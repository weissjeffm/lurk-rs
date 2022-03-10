#![allow(non_snake_case)]

use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use merlin::Transcript;
use nova::{
    bellperson::{
        r1cs::{NovaShape, NovaWitness},
        shape_cs::ShapeCS,
        solver::SatisfyingAssignment,
    },
    errors::NovaError,
    r1cs::{
        R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
    },
    traits::Group,
    FinalSNARK, StepSNARK,
};
use once_cell::sync::Lazy;
use pasta_curves::pallas;

use crate::circuit::CircuitFrame;
use crate::eval::{Evaluator, Frame, Witness, IO};

use crate::proof::Provable;
use crate::store::{Ptr, Store};

type PallasPoint = pallas::Point;
type PallasScalar = pallas::Scalar;

static EMPTY_STORE: Lazy<Store<PallasScalar>> = Lazy::new(Store::<PallasScalar>::default);
static BLANK_CIRCUIT_FRAME: Lazy<
    CircuitFrame<'_, PallasScalar, IO<PallasScalar>, Witness<PallasScalar>>,
> = Lazy::new(|| CircuitFrame::blank(&EMPTY_STORE));

pub struct Proof<G: Group> {
    pub step_proofs: Vec<StepSNARK<G>>,
    pub final_proof: FinalSNARK<G>,
    pub final_instance: RelaxedR1CSInstance<G>,
}

impl<G: Group> Proof<G> {
    pub fn verify(
        &self,
        shape_and_gens: &(R1CSShape<G>, R1CSGens<G>),
        instance: &RelaxedR1CSInstance<G>,
    ) -> bool {
        self.final_proof
            .verify(&shape_and_gens.1, &shape_and_gens.0, instance)
            .is_ok()
    }
}

pub trait Nova
where
    <Self::Grp as Group>::Scalar: ff::PrimeField,
{
    type Grp: Group;

    fn make_r1cs(
        circuit_frame: CircuitFrame<
            '_,
            <Self::Grp as Group>::Scalar,
            IO<<Self::Grp as Group>::Scalar>,
            Witness<<Self::Grp as Group>::Scalar>,
        >,
        shape: &R1CSShape<Self::Grp>,
        gens: &R1CSGens<Self::Grp>,
    ) -> Result<(R1CSInstance<Self::Grp>, R1CSWitness<Self::Grp>), NovaError> {
        let mut cs = SatisfyingAssignment::<Self::Grp>::new();

        circuit_frame.synthesize(&mut cs).unwrap();

        let (instance, witness) = cs.r1cs_instance_and_witness(shape, gens)?;

        Ok((instance, witness))
    }

    fn evaluate_and_prove(
        expr: Ptr<<Self::Grp as Group>::Scalar>,
        env: Ptr<<Self::Grp as Group>::Scalar>,
        store: &mut Store<<Self::Grp as Group>::Scalar>,
        limit: usize,
    ) -> Result<(Proof<Self::Grp>, RelaxedR1CSInstance<Self::Grp>), SynthesisError> {
        let mut evaluator = Evaluator::new(expr, env, store, limit);
        let frames = evaluator.iter().collect::<Vec<_>>();
        let (shape, gens) = Self::make_shape_and_gens();

        store.hydrate_scalar_cache();

        Self::make_proof(frames.as_slice(), &shape, &gens, store, true)
    }

    fn make_shape_and_gens() -> (R1CSShape<Self::Grp>, R1CSGens<Self::Grp>);

    fn make_proof(
        frames: &[Frame<IO<<Self::Grp as Group>::Scalar>, Witness<<Self::Grp as Group>::Scalar>>],
        shape: &R1CSShape<Self::Grp>,
        gens: &R1CSGens<Self::Grp>,
        store: &mut Store<<Self::Grp as Group>::Scalar>,
        verify_steps: bool, // Sanity check for development, until we have recursion.
    ) -> Result<(Proof<Self::Grp>, RelaxedR1CSInstance<Self::Grp>), SynthesisError> {
        let r1cs_instances = frames
            .iter()
            .map(|f| {
                let circuit_frame = CircuitFrame::from_frame(f.clone(), store);
                dbg!(&circuit_frame.public_inputs(store));

                Self::make_r1cs(circuit_frame, shape, gens).unwrap()
            })
            .collect::<Vec<_>>();

        assert!(r1cs_instances.len() > 1);

        let mut step_proofs = Vec::new();
        let mut prover_transcript = Transcript::new(b"LurkProver");
        let mut verifier_transcript = Transcript::new(b"LurkVerifier");

        let initial_acc = (
            RelaxedR1CSInstance::default(gens, shape),
            RelaxedR1CSWitness::default(shape),
        );

        let (acc_U, acc_W) =
            r1cs_instances
                .iter()
                .fold(initial_acc, |(acc_U, acc_W), (next_U, next_W)| {
                    let (step_proof, (step_U, step_W)) = Self::make_step_snark(
                        gens,
                        shape,
                        &acc_U,
                        &acc_W,
                        next_U,
                        next_W,
                        &mut prover_transcript,
                    );
                    if verify_steps {
                        step_proof
                            .verify(&acc_U, next_U, &mut verifier_transcript)
                            .unwrap();
                        step_proofs.push(step_proof);
                    };
                    (step_U, step_W)
                });

        let final_proof = Self::make_final_snark(&acc_W);

        let proof = Proof {
            step_proofs,
            final_proof,
            final_instance: acc_U.clone(),
        };

        Ok((proof, acc_U))
    }

    fn make_step_snark(
        gens: &R1CSGens<Self::Grp>,
        S: &R1CSShape<Self::Grp>,
        r_U: &RelaxedR1CSInstance<Self::Grp>,
        r_W: &RelaxedR1CSWitness<Self::Grp>,
        U2: &R1CSInstance<Self::Grp>,
        W2: &R1CSWitness<Self::Grp>,
        prover_transcript: &mut merlin::Transcript,
    ) -> (
        StepSNARK<Self::Grp>,
        (
            RelaxedR1CSInstance<Self::Grp>,
            RelaxedR1CSWitness<Self::Grp>,
        ),
    ) {
        let res = StepSNARK::prove(gens, S, r_U, r_W, U2, W2, prover_transcript);
        res.expect("make_step_snark failed")
    }

    fn make_final_snark(W: &RelaxedR1CSWitness<Self::Grp>) -> FinalSNARK<Self::Grp> {
        // produce a final SNARK
        let res = FinalSNARK::prove(W);
        res.expect("make_final_snark failed")
    }
}

pub struct NovaProver();
impl Nova for NovaProver {
    type Grp = PallasPoint;

    fn make_shape_and_gens() -> (R1CSShape<Self::Grp>, R1CSGens<Self::Grp>) {
        let mut cs = ShapeCS::<Self::Grp>::new();
        BLANK_CIRCUIT_FRAME.clone().synthesize(&mut cs).unwrap();

        let shape = cs.r1cs_shape();
        let gens = cs.r1cs_gens();

        (shape, gens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::empty_sym_env;
    use crate::proof::verify_sequential_css;
    use crate::proof::SequentialCS;

    use bellperson::util_cs::{metric_cs::MetricCS, Comparable, Delta};
    use pallas::Scalar as Fr;

    const DEFAULT_CHECK_GROTH16: bool = false;

    fn outer_prove_aux<Fo: Fn(&'_ mut Store<Fr>) -> Ptr<Fr>>(
        source: &str,
        expected_result: Fo,
        expected_iterations: usize,
        check_nova: bool,
        check_constraint_systems: bool,
        limit: usize,
        debug: bool,
    ) {
        let mut s = Store::default();
        let expected_result = expected_result(&mut s);

        let expr = s.read(source).unwrap();

        let e = empty_sym_env(&s);

        let proof_results = if check_nova {
            Some(
                NovaProver::evaluate_and_prove(expr.clone(), empty_sym_env(&s), &mut s, limit)
                    .unwrap(),
            )
        } else {
            None
        };

        let shape_and_gens = NovaProver::make_shape_and_gens();
        if check_nova {
            if let Some((proof, instance)) = proof_results {
                proof.verify(&shape_and_gens, &instance);
            }
        }

        let constraint_systems = if check_constraint_systems {
            Some(CircuitFrame::outer_synthesize(expr, e, &mut s, limit, false).unwrap())
        } else {
            None
        };

        if let Some(cs) = constraint_systems {
            if !debug {
                assert_eq!(expected_iterations, cs.len());
                assert_eq!(expected_result, cs[cs.len() - 1].0.output.expr);
            }
            let constraint_systems_verified = verify_sequential_css::<Fr>(&cs, &s).unwrap();
            assert!(constraint_systems_verified);

            check_cs_deltas(&cs, limit);
        };
    }

    pub fn check_cs_deltas(
        constraint_systems: &SequentialCS<Fr, IO<Fr>, Witness<Fr>>,
        limit: usize,
    ) -> () {
        let mut cs_blank = MetricCS::<Fr>::new();
        let store = Store::<Fr>::default();
        let blank_frame = CircuitFrame::<Fr, _, _>::blank(&store);
        blank_frame
            .synthesize(&mut cs_blank)
            .expect("failed to synthesize");

        for (i, (_frame, cs)) in constraint_systems.iter().take(limit).enumerate() {
            let delta = cs.delta(&cs_blank, true);
            dbg!(i, &delta);
            assert!(delta == Delta::Equal);
        }
    }

    #[test]
    fn outer_prove_arithmetic_let() {
        outer_prove_aux(
            &"(let ((a 5)
                     (b 1)
                     (c 2))
                (/ (+ a b) c))",
            |store| store.num(3),
            18,
            true, // Always check Nova in at least one test.
            true,
            100,
            false,
        );
    }

    #[test]
    fn outer_prove_binop() {
        outer_prove_aux(
            &"(+ 1 2)",
            |store| store.num(3),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            100,
            false,
        );
    }

    #[test]
    fn outer_prove_eq() {
        outer_prove_aux(
            &"(eq 5 5)",
            |store| store.t(),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            100,
            false,
        );

        // outer_prove_aux(&"(eq 5 6)", Expression::Nil, 5, false, true, 100, false);
    }

    #[test]
    fn outer_prove_num_equal() {
        outer_prove_aux(
            &"(= 5 5)",
            |store| store.t(),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            100,
            false,
        );
        outer_prove_aux(
            &"(= 5 6)",
            |store| store.nil(),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            100,
            false,
        );
    }

    #[test]
    fn outer_prove_if() {
        outer_prove_aux(
            &"(if t 5 6)",
            |store| store.num(5),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            100,
            false,
        );

        outer_prove_aux(
            &"(if t 5 6)",
            |store| store.num(5),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            100,
            false,
        )
    }
    #[test]
    fn outer_prove_if_fully_evaluates() {
        outer_prove_aux(
            &"(if t (+ 5 5) 6)",
            |store| store.num(10),
            5,
            DEFAULT_CHECK_GROTH16,
            true,
            100,
            false,
        );
    }

    #[test]
    #[ignore] // Skip expensive tests in CI for now. Do run these locally, please.
    fn outer_prove_recursion1() {
        outer_prove_aux(
            &"(letrec* ((exp (lambda (base)
                               (lambda (exponent)
                                 (if (= 0 exponent)
                                     1
                                     (* base ((exp base) (- exponent 1))))))))
                ((exp 5) 3))",
            |store| store.num(125),
            // 117, // FIXME: is this change correct?
            91,
            DEFAULT_CHECK_GROTH16,
            true,
            200,
            false,
        );
    }

    #[test]
    #[ignore] // Skip expensive tests in CI for now. Do run these locally, please.
    fn outer_prove_recursion2() {
        outer_prove_aux(
            &"(letrec* ((exp (lambda (base)
                                  (lambda (exponent)
                                     (lambda (acc)
                                       (if (= 0 exponent)
                                          acc
                                          (((exp base) (- exponent 1)) (* acc base))))))))
                (((exp 5) 5) 1))",
            |store| store.num(3125),
            // 248, // FIXME: is this change correct?
            201,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate() {
        outer_prove_aux(
            &"((lambda (x) x) 99)",
            |store| store.num(99),
            4,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate2() {
        outer_prove_aux(
            &"((lambda (y)
                   ((lambda (x) y) 888))
                 99)",
            |store| store.num(99),
            9,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate3() {
        outer_prove_aux(
            &"((lambda (y)
                    ((lambda (x)
                       ((lambda (z) z)
                        x))
                     y))
                  999)",
            |store| store.num(999),
            10,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate4() {
        outer_prove_aux(
            &"((lambda (y)
                    ((lambda (x)
                       ((lambda (z) z)
                        x))
                     ;; NOTE: We pass a different value here.
                     888))
                  999)",
            |store| store.num(888),
            10,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate5() {
        outer_prove_aux(
            &"(((lambda (fn)
                     (lambda (x) (fn x)))
                   (lambda (y) y))
                  999)",
            |store| store.num(999),
            13,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_sum() {
        outer_prove_aux(
            &"(+ 2 (+ 3 4))",
            |store| store.num(9),
            6,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_diff() {
        outer_prove_aux(
            &"(- 9 5)",
            |store| store.num(4),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_product() {
        outer_prove_aux(
            &"(* 9 5)",
            |store| store.num(45),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_quotient() {
        outer_prove_aux(
            &"(/ 21 3)",
            |store| store.num(7),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_equal1() {
        outer_prove_aux(
            &"(= 5 5)",
            |store| store.t(),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_equal2() {
        outer_prove_aux(
            &"(((lambda (x)
                   (lambda (y)
                     (+ x y)))
                 2)
                3)",
            |store| store.num(5),
            13,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_let_simple() {
        outer_prove_aux(
            &"(let ((a 1))
                 a)",
            |store| store.num(1),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_let_bug() {
        //TODO: fix this test
        outer_prove_aux(
            &"(let () (+ 1 2))",
            |store| store.num(3),
            4,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_let() {
        outer_prove_aux(
            &"(let ((a 1)
                      (b 2)
                      (c 3))
                 (+ a (+ b c)))",
            |store| store.num(6),
            18,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_arithmetic() {
        outer_prove_aux(
            &"((((lambda (x)
                     (lambda (y)
                       (lambda (z)
                         (* z
                            (+ x y)))))
                   2)
                  3)
                 4)",
            |store| store.num(20),
            23,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_arithmetic_let() {
        outer_prove_aux(
            &"(let ((x 2)
                       (y 3)
                       (z 4))
                  (* z (+ x y)))",
            |store| store.num(20),
            18,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_comparison() {
        outer_prove_aux(
            &"(let ((x 2)
                      (y 3)
                      (z 4))
                 (= 20 (* z
                          (+ x y))))",
            |store| store.t(),
            21,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_conditional() {
        outer_prove_aux(
            &"(let ((true (lambda (a)
                              (lambda (b)
                                a)))
                      (false (lambda (a)
                               (lambda (b)
                                 b)))
                      ;; NOTE: We cannot shadow IF because it is built-in.
                      (if- (lambda (a)
                             (lambda (c)
                               (lambda (cond)
                                 ((cond a) c))))))
                 (((if- 5) 6) true))",
            |store| store.num(5),
            35,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_conditional2() {
        outer_prove_aux(
            &"(let ((true (lambda (a)
                              (lambda (b)
                                a)))
                      (false (lambda (a)
                               (lambda (b)
                                 b)))
                      ;; NOTE: We cannot shadow IF because it is built-in.
                      (if- (lambda (a)
                             (lambda (c)
                               (lambda (cond)
                                 ((cond a) c))))))
                 (((if- 5) 6) false))",
            |store| store.num(6),
            32,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_fundamental_conditional_bug() {
        outer_prove_aux(
            &"(let ((true (lambda (a)
                              (lambda (b)
                                a)))
                      ;; NOTE: We cannot shadow IF because it is built-in.
                      (if- (lambda (a)
                             (lambda (c)
                               (lambda (cond)
                                 ((cond a) c))))))
                 (((if- 5) 6) true))",
            |store| store.num(5),
            32,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_if() {
        outer_prove_aux(
            &"(if nil 5 6)",
            |store| store.num(6),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_fully_evaluates() {
        outer_prove_aux(
            &"(if t (+ 5 5) 6)",
            |store| store.num(10),
            5,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_recursion() {
        outer_prove_aux(
            &"(letrec ((exp (lambda (base)
                                  (lambda (exponent)
                                    (if (= 0 exponent)
                                        1
                                        (* base ((exp base) (- exponent 1))))))))
                          ((exp 5) 3))",
            |store| store.num(125),
            91,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_recursion_multiarg() {
        outer_prove_aux(
            &"(letrec ((exp (lambda (base exponent)
                                  (if (= 0 exponent)
                                      1
                                      (* base (exp base (- exponent 1)))))))
                          (exp 5 3))",
            |store| store.num(125),
            95,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_recursion_optimized() {
        outer_prove_aux(
            &"(let ((exp (lambda (base)
                               (letrec ((base-inner
                                          (lambda (exponent)
                                            (if (= 0 exponent)
                                                1
                                                (* base (base-inner (- exponent 1)))))))
                                        base-inner))))
                   ((exp 5) 3))",
            |store| store.num(125),
            75,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_tail_recursion() {
        //TODO: Fix problem (process didn't exit successfully ... (signal: 9, SIGKILL: kill))
        outer_prove_aux(
            &"(letrec ((exp (lambda (base)
                                  (lambda (exponent-remaining)
                                    (lambda (acc)
                                      (if (= 0 exponent-remaining)
                                          acc
                                          (((exp base) (- exponent-remaining 1)) (* acc base))))))))
                          (((exp 5) 3) 1))",
            |store| store.num(125),
            129,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_tail_recursion_somewhat_optimized() {
        outer_prove_aux(
            &"(letrec ((exp (lambda (base)
                                  (letrec ((base-inner
                                             (lambda (exponent-remaining)
                                               (lambda (acc)
                                                 (if (= 0 exponent-remaining)
                                                     acc
                                                     ((base-inner (- exponent-remaining 1)) (* acc base)))))))
                                           base-inner))))
                          (((exp 5) 3) 1))",
            |store| store.num(125),
            110,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_no_mutual_recursion() {
        outer_prove_aux(
            &"(letrec ((even (lambda (n)
                                 (if (= 0 n)
                                     t
                                     (odd (- n 1)))))
                         (odd (lambda (n)
                                (even (- n 1)))))
                        ;; NOTE: This is not true mutual-recursion.
                        ;; However, it exercises the behavior of LETREC.
                        (odd 1))",
            |store| store.t(),
            22,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_let_no_body() {
        outer_prove_aux(
            &"(let ((a 9)))",
            |store| store.nil(),
            3,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_cons1() {
        outer_prove_aux(
            &"(car (cons 1 2))",
            |store| store.num(1),
            5,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }

    #[test]
    #[ignore]
    fn outer_prove_evaluate_cons2() {
        outer_prove_aux(
            &"(cdr (cons 1 2))",
            |store| store.num(2),
            5,
            DEFAULT_CHECK_GROTH16,
            true,
            300,
            false,
        );
    }
}

/*
;; outer-evaluate-no-mutual-recursion
!(:assert-eq t (letrec ((even (lambda (n)
                                 (if (= 0 n)
                                     t
                                     (odd (- n 1)))))
                         (odd (lambda (n)
                                (even (- n 1)))))
                        ;; NOTE: This is not true mutual-recursion.
                        ;; However, it exercises the behavior of LETREC.
                        (odd 1)))

;; outer-evaluate-no-mutual-recursion
!(:assert-error (letrec ((even (lambda (n)
                                  (if (= 0 n)
                                      t
                                      (odd (- n 1)))))
                          (odd (lambda (n)
                                 (even (- n 1)))))
                         ;; NOTE: This is not true mutual-recursion.
                         ;; However, it exercises the behavior of LETREC.
                         (odd 2)))

;; outer-evaluate-let-scope

!(:assert-error (let ((closure (lambda (x)
                                 ;; This use of CLOSURE is unbound.
                                 closure)))
                  (closure 1)))

;; outer-evaluate-let-no-body
!(:assert-eq nil (let ((a 9))))

;; outer-evaluate-cons 1
!(:assert-eq 1 (car (cons 1 2)))

;; outer-evaluate-cons 2
!(:assert-eq 2 (cdr (cons 1 2)))

;; outer-evaluate-cons-in-function 1
!(:assert-eq 2 (((lambda (a)
                   (lambda (b)
                     (car (cons a b))))
                 2)
                3))

;; outer-evaluate-cons-in-function 2
!(:assert-eq 3 (((lambda (a)
                   (lambda (b)
                     (cdr (cons a b))))
                 2)
                3))

;; multiarg-eval-bug
!(:assert-eq 2 (car (cdr '(1 2 3 4))))

;; outer-evaluate-zero-arg-lambda 1
!(:assert-eq 123 ((lambda () 123)))

;; outer-evaluate-zero-arg-lambda 2
!(:assert-eq 10 (let ((x 9) (f (lambda () (+ x 1)))) (f)))

;; minimal-tail-call
!(:assert-eq 123 (letrec
                  ((f (lambda (x)
                        (if (= x 140)
                            123
                            (f (+ x 1))))))
                  (f 0)))

;; multiple-letrec-bindings
!(:assert-eq 123 (letrec
                  ((x 888)
                   (f (lambda (x)
                        (if (= x 5)
                            123
                            (f (+ x 1))))))
                  (f 0)))

;; tail-call2
!(:assert-eq 123 (letrec
                  ((f (lambda (x)
                        (if (= x 5)
                            123
                            (f (+ x 1)))))
                   (g (lambda (x) (f x))))
                  (g 0)))

;; outer-evaluate-make-tree
!(:assert-eq '(((h . g) . (f . e)) . ((d . c) . (b . a)))
             (letrec ((mapcar (lambda (f list)
                                 (if (eq list nil)
                                     nil
                                     (cons (f (car list)) (mapcar f (cdr list))))))
                       (make-row (lambda (list)
                                   (if (eq list nil)
                                       nil
                                       (let ((cdr (cdr list)))
                                         (cons (cons (car list) (car cdr))
                                               (make-row (cdr cdr)))))))
                       (make-tree-aux (lambda (list)
                                        (let ((row (make-row list)))
                                          (if (eq (cdr row) nil)
                                              row
                                              (make-tree-aux row)))))
                       (make-tree (lambda (list)
                                    (car (make-tree-aux list))))
                       (reverse-tree (lambda (tree)
                                       (if (atom tree)
                                           tree
                                           (cons (reverse-tree (cdr tree))
                                                 (reverse-tree (car tree)))))))
                      (reverse-tree
                       (make-tree '(a b c d e f g h)))))

;; outer-evaluate-multiple-letrecstar-bindings
!(:assert-eq 13 (letrec ((double (lambda (x) (* 2 x)))
                          (square (lambda (x) (* x x))))
                         (+ (square 3) (double 2))))

;; outer-evaluate-multiple-letrecstar-bindings-referencing
!(:assert-eq 11 (letrec ((double (lambda (x) (* 2 x)))
                          (double-inc (lambda (x) (+ 1 (double x)))))
                         (+ (double 3) (double-inc 2))))

;; outer-evaluate-multiple-letrecstar-bindings-recursive
!(:assert-eq 33 (letrec ((exp (lambda (base exponent)
                                 (if (= 0 exponent)
                                     1
                                     (* base (exp base (- exponent 1))))))
                          (exp2 (lambda (base exponent)
                                  (if (= 0 exponent)
                                      1
                                      (* base (exp2 base (- exponent 1))))))
                          (exp3 (lambda (base exponent)
                                  (if (= 0 exponent)
                                      1
                                      (* base (exp3 base (- exponent 1)))))))
                         (+ (+ (exp 3 2) (exp2 2 3))
                            (exp3 4 2))))

;; dont-discard-rest-env
!(:assert-eq 18 (let ((z 9))
                  (letrec ((a 1)
                            (b 2)
                            (l (lambda (x) (+ z x))))
                           (l 9))))

;; let-restore-saved-env
!(:assert-error (+ (let ((a 1)) a) a))

;; let-restore-saved-env2
!(:assert-error (+ (let ((a 1) (a 2)) a) a))

;; letrec-restore-saved-env
!(:assert-error (+ (letrec ((a 1)(a 2)) a) a))

;; lookup-restore-saved-env
!(:assert-error (+ (let ((a 1))
                     a)
                   a))

;; tail-call-restore-saved-env
!(:assert-error (let ((outer (letrec
                               ((x 888)
                                (f (lambda (x)
                                     (if (= x 2)
                                         123
                                         (f (+ x 1))))))
                               f)))
                  ;; This should be an error. X should not be bound here.
                  (+ (outer 0) x)))

;; binop-restore-saved-env
!(:assert-error (let ((outer (let ((f (lambda (x)
                                          (+ (let ((a 9)) a) x))))
                                f)))
                  ;; This should be an error. X should not be bound here.
                  (+ (outer 1) x)))
*/
