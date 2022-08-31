use crate::field::LurkField;
use std::collections::BTreeMap;
use std::hash::Hash;

use crate::store::{
    ContTag, Op1, Op2, Pointer, Ptr, Rel2, ScalarContPtr, ScalarPointer, ScalarPtr, Store, Tag,
};
use crate::Num;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UPtr<F: LurkField>(F, F);

impl<F: LurkField> Copy for UPtr<F> {}

impl<F: LurkField> PartialOrd for UPtr<F> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        (self.0.to_repr().as_ref(), self.1.to_repr().as_ref())
            .partial_cmp(&(other.0.to_repr().as_ref(), other.1.to_repr().as_ref()))
    }
}

impl<F: LurkField> Ord for UPtr<F> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.0.to_repr().as_ref(), self.1.to_repr().as_ref())
            .cmp(&(other.0.to_repr().as_ref(), other.1.to_repr().as_ref()))
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<F: LurkField> Hash for UPtr<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_repr().as_ref().hash(state);
        self.1.to_repr().as_ref().hash(state);
    }
}

impl<F: LurkField> ScalarPointer<F> for UPtr<F> {
    fn from_parts(tag: F, value: F) -> Self {
        UPtr(tag, value)
    }

    fn tag(&self) -> &F {
        &self.0
    }

    fn value(&self) -> &F {
        &self.1
    }
}

impl<F: LurkField> From<ScalarPtr<F>> for UPtr<F> {
    fn from(x: ScalarPtr<F>) -> Self {
        UPtr(*x.tag(), *x.value())
    }
}

impl<F: LurkField> From<ScalarContPtr<F>> for UPtr<F> {
    fn from(x: ScalarContPtr<F>) -> Self {
        UPtr(*x.tag(), *x.value())
    }
}

pub enum UTag {
    Tag(Tag),
    ContTag(ContTag),
}

impl UTag {
    pub fn from_field<F: LurkField>(f: F) -> Option<Self> {
        if let Some(tag) = Tag::from_field(f) {
            Some(UTag::Tag(tag))
        } else {
            ContTag::from_field(f).map(UTag::ContTag)
        }
    }

    fn arity(self) -> usize {
        match self {
            Self::Tag(Tag::Nil) => todo!(),
            Self::Tag(Tag::Cons) => 4,
            Self::Tag(Tag::Comm) => 3,
            Self::Tag(Tag::Sym) => todo!(),
            Self::Tag(Tag::Fun) => 6,
            Self::Tag(Tag::Str) => todo!(),
            Self::Tag(Tag::Thunk) => 4,
            Self::Tag(Tag::Char) => 1,
            Self::Tag(Tag::Num) => 1,
            Self::ContTag(ContTag::Outermost) => todo!(),
            Self::ContTag(ContTag::Call0) => 6,
            Self::ContTag(ContTag::Call) => 6,
            Self::ContTag(ContTag::Call2) => 6,
            Self::ContTag(ContTag::Tail) => 4,
            Self::ContTag(ContTag::Error) => todo!(),
            Self::ContTag(ContTag::Lookup) => 4,
            Self::ContTag(ContTag::Unop) => 3,
            Self::ContTag(ContTag::Binop) => 7,
            Self::ContTag(ContTag::Binop2) => 5,
            Self::ContTag(ContTag::Relop) => 7,
            Self::ContTag(ContTag::Relop2) => 5,
            Self::ContTag(ContTag::If) => 4,
            Self::ContTag(ContTag::Let) => 8,
            Self::ContTag(ContTag::LetRec) => 8,
            Self::ContTag(ContTag::Emit) => 2,
            Self::ContTag(ContTag::Dummy) => todo!(),
            Self::ContTag(ContTag::Terminal) => todo!(),
        }
    }
}

/// `ScalarStore` allows realization of a graph of `ScalarPtr`s suitable for serialization to IPLD. `ScalarExpression`s
/// are composed only of `ScalarPtr`s, so `scalar_map` suffices to allow traverseing an arbitrary DAG.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarStore<F: LurkField> {
    scalar_map: BTreeMap<ScalarPtr<F>, Option<ScalarExpression<F>>>,
    scalar_cont_map: BTreeMap<ScalarContPtr<F>, Option<ScalarContinuation<F>>>,
}

impl<'a, F: LurkField> ScalarStore<F> {
    /// Create a new `ScalarStore` and add all `ScalarPtr`s reachable in the scalar representation of `expr`.
    pub fn new_with_expr(store: &Store<F>, expr: &Ptr<F>) -> (Self, Option<ScalarPtr<F>>) {
        let mut new = Self::default();
        let mut pending = Vec::new();
        let scalar_ptr = new.add_one_ptr(&mut pending, store, expr);
        if let Some(scalar_ptr) = scalar_ptr {
            (new, Some(scalar_ptr))
        } else {
            (new, None)
        }
    }

    /// Add all ScalarPtrs representing and reachable from expr.
    pub fn add_one_ptr(
        &mut self,
        pending: &mut Vec<ScalarPtr<F>>,
        store: &Store<F>,
        expr: &Ptr<F>,
    ) -> Option<ScalarPtr<F>> {
        let scalar_ptr = self.add_ptr(pending, store, expr);
        self.finalize(pending, store);
        scalar_ptr
    }

    /// Add the `ScalarPtr` representing `expr`, and queue it for proceessing.
    pub fn add_ptr(
        &mut self,
        pending: &mut Vec<ScalarPtr<F>>,
        store: &Store<F>,
        expr: &Ptr<F>,
    ) -> Option<ScalarPtr<F>> {
        // Find the scalar_ptr representing ptr.
        if let Some(scalar_ptr) = store.get_expr_hash(expr) {
            self.add(pending, store, expr, scalar_ptr);
            Some(scalar_ptr)
        } else {
            None
        }
    }

    /// Add a single `ScalarPtr` and queue it for processing.
    /// NOTE: This requires that `store.scalar_cache` has been hydrated.
    fn add_scalar_ptr(
        &mut self,
        pending: &mut Vec<ScalarPtr<F>>,
        store: &Store<F>,
        scalar_ptr: ScalarPtr<F>,
    ) {
        // Find the ptr corresponding to scalar_ptr.
        if let Some(ptr) = store.scalar_ptr_map.get(&scalar_ptr) {
            self.add(pending, store, &*ptr, scalar_ptr);
        }
    }

    /// Add the `ScalarPtr` and `ScalarExpression` associated with `ptr`. The relationship between `ptr` and
    /// `scalar_ptr` is not checked here, so `add` should only be called by `add_ptr` and `add_scalar_ptr`, which
    /// enforce this relationship.
    fn add(
        &mut self,
        pending: &mut Vec<ScalarPtr<F>>,
        store: &Store<F>,
        ptr: &Ptr<F>,
        scalar_ptr: ScalarPtr<F>,
    ) {
        let mut new_pending_scalar_ptrs: Vec<ScalarPtr<F>> = Default::default();

        // If `scalar_ptr` is not already in the map, queue its children for processing.
        self.scalar_map.entry(scalar_ptr).or_insert_with(|| {
            let scalar_expression = ScalarExpression::from_ptr(store, ptr)?;
            if let Some(more_scalar_ptrs) = Self::child_scalar_ptrs(&scalar_expression) {
                new_pending_scalar_ptrs.extend(more_scalar_ptrs);
            }
            Some(scalar_expression)
        });

        pending.extend(new_pending_scalar_ptrs);
    }

    /// All the `ScalarPtr`s directly reachable from `scalar_expression`, if any.
    fn child_scalar_ptrs(scalar_expression: &ScalarExpression<F>) -> Option<Vec<ScalarPtr<F>>> {
        match scalar_expression {
            ScalarExpression::Nil => None,
            ScalarExpression::Cons(car, cdr) => Some([*car, *cdr].into()),
            ScalarExpression::Comm(_, payload) => Some([*payload].into()),
            ScalarExpression::Sym(_str) => None,
            ScalarExpression::Fun {
                arg,
                body,
                closed_env,
            } => Some([*arg, *body, *closed_env].into()),
            ScalarExpression::Num(_) => None,
            ScalarExpression::Str(_) => None,
            ScalarExpression::Thunk(_) => None,
            ScalarExpression::Char(_) => None,
        }
    }

    /// Unqueue all the pending `ScalarPtr`s and add them, queueing all of their children, then repeat until the queue
    /// is pending queue is empty.
    fn add_pending_scalar_ptrs(&mut self, pending: &mut Vec<ScalarPtr<F>>, store: &Store<F>) {
        while let Some(scalar_ptr) = pending.pop() {
            self.add_scalar_ptr(pending, store, scalar_ptr);
        }
    }

    /// Method which finalizes the `ScalarStore`, ensuring that all reachable `ScalarPtr`s have been added.
    pub fn finalize(&mut self, pending: &mut Vec<ScalarPtr<F>>, store: &Store<F>) {
        self.add_pending_scalar_ptrs(pending, store);
    }
    pub fn get_expr(&self, ptr: &ScalarPtr<F>) -> Option<&ScalarExpression<F>> {
        let x = self.scalar_map.get(ptr)?;
        (*x).as_ref()
    }

    pub fn get_cont(&self, ptr: &ScalarContPtr<F>) -> Option<&ScalarContinuation<F>> {
        let x = self.scalar_cont_map.get(ptr)?;
        (*x).as_ref()
    }

    pub fn to_store_with_expr(&mut self, ptr: &ScalarPtr<F>) -> Option<(Store<F>, Ptr<F>)> {
        let mut store = Store::new();

        let ptr = store.intern_scalar_ptr(*ptr, self)?;

        for scalar_ptr in self.scalar_map.keys() {
            store.intern_scalar_ptr(*scalar_ptr, self);
        }
        for ptr in self.scalar_cont_map.keys() {
            store.intern_scalar_cont_ptr(*ptr, self);
        }
        Some((store, ptr))
    }
    pub fn to_store(&mut self) -> Option<Store<F>> {
        let mut store = Store::new();

        for ptr in self.scalar_map.keys() {
            store.intern_scalar_ptr(*ptr, self);
        }
        for ptr in self.scalar_cont_map.keys() {
            store.intern_scalar_cont_ptr(*ptr, self);
        }
        Some(store)
    }

    pub fn ser_f(self) -> Vec<F> {
        let mut merged_map: BTreeMap<UPtr<F>, Vec<F>> = BTreeMap::new();
        for (ptr, expr) in self.scalar_map {
            let expr: Vec<F> = expr.map_or_else(|| vec![F::zero()], |x| x.ser_f());
            merged_map.insert(ptr.into(), expr);
        }
        for (ptr, cont) in self.scalar_cont_map {
            let cont: Vec<F> = cont.map_or_else(|| vec![F::zero()], |x| x.ser_f());
            merged_map.insert(ptr.into(), cont);
        }
        let mut res = Vec::new();
        for (UPtr(tag, dig), mut vec) in merged_map.into_iter() {
            res.push(tag);
            res.push(dig);
            res.append(&mut vec);
        }
        res
    }

    pub fn de_f(stream: &Vec<F>) -> Result<Self, ()> {
        let mut idx: usize = 0;
        let len = stream.len();
        let mut scalar_map: BTreeMap<ScalarPtr<F>, Option<ScalarExpression<F>>> = BTreeMap::new();
        let mut scalar_cont_map: BTreeMap<ScalarContPtr<F>, Option<ScalarContinuation<F>>> =
            BTreeMap::new();

        while idx < len {
            let utag = UTag::from_field(stream[idx]).map_or_else(|| Err(()), Ok)?;
            let value = stream[idx + 1];
            match utag {
                UTag::Tag(tag) => {
                    let ptr = ScalarPtr::from_parts(tag.as_field(), value);
                    let (idx2, expr) = ScalarExpression::de_f(stream, idx + 2, tag)?;
                    scalar_map.insert(ptr, Some(expr));
                    idx = idx2
                }
                UTag::ContTag(tag) => {
                    let ptr = ScalarContPtr::from_parts(tag.as_field(), value);
                    let (idx2, expr) = ScalarContinuation::de_f(stream, idx + 2, tag)?;
                    scalar_cont_map.insert(ptr, Some(expr));
                    idx = idx2
                }
            }
        }

        Ok(ScalarStore {
            scalar_map,
            scalar_cont_map,
        })
    }
}

impl<'a, F: LurkField> ScalarExpression<F> {
    fn from_ptr(store: &Store<F>, ptr: &Ptr<F>) -> Option<Self> {
        match ptr.tag() {
            Tag::Nil => Some(ScalarExpression::Nil),
            Tag::Cons => store.fetch_cons(ptr).and_then(|(car, cdr)| {
                store.get_expr_hash(car).and_then(|car| {
                    store
                        .get_expr_hash(cdr)
                        .map(|cdr| ScalarExpression::Cons(car, cdr))
                })
            }),
            Tag::Comm => store.fetch_comm(ptr).and_then(|(secret, payload)| {
                store
                    .get_expr_hash(payload)
                    .map(|payload| ScalarExpression::Comm(secret.0, payload))
            }),
            Tag::Sym => store
                .fetch_sym(ptr)
                .map(|str| ScalarExpression::Sym(str.into())),
            Tag::Fun => store.fetch_fun(ptr).and_then(|(arg, body, closed_env)| {
                store.get_expr_hash(arg).and_then(|arg| {
                    store.get_expr_hash(body).and_then(|body| {
                        store
                            .get_expr_hash(closed_env)
                            .map(|closed_env| ScalarExpression::Fun {
                                arg,
                                body,
                                closed_env,
                            })
                    })
                })
            }),
            Tag::Num => store.fetch_num(ptr).map(|num| match num {
                Num::U64(x) => ScalarExpression::Num((*x).into()),
                Num::Scalar(x) => ScalarExpression::Num(*x),
            }),

            Tag::Str => store
                .fetch_str(ptr)
                .map(|str| ScalarExpression::Str(str.to_string())),
            Tag::Char => store.fetch_char(ptr).map(ScalarExpression::Char),
            Tag::Thunk => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarExpression<F: LurkField> {
    Nil,
    Cons(ScalarPtr<F>, ScalarPtr<F>),
    Comm(F, ScalarPtr<F>),
    Sym(String),
    Fun {
        arg: ScalarPtr<F>,
        body: ScalarPtr<F>,
        closed_env: ScalarPtr<F>,
    },
    Num(F),
    Str(String),
    Thunk(ScalarThunk<F>),
    Char(char),
}

impl<'a, F: LurkField> Default for ScalarExpression<F> {
    fn default() -> Self {
        Self::Nil
    }
}

pub fn small_string_to_f<F: LurkField>(s: String) -> Option<F> {
    let bytes = s.as_bytes();
    if bytes.len() as u32 > F::CAPACITY {
        None
    } else {
        let mut def: F::Repr = F::default().to_repr();
        def.as_mut().copy_from_slice(bytes);
        F::from_repr(def).into()
    }
}

pub fn char_to_f<F: LurkField>(c: char) -> Option<F> {
    let bytes: Vec<u8> = (c as u32).to_be_bytes().into();
    let mut def: F::Repr = F::default().to_repr();
    def.as_mut().copy_from_slice(&bytes);
    F::from_repr(def).into()
}

impl<F: LurkField> ScalarExpression<F> {
    pub fn ser_f(self) -> Vec<F> {
        match self {
            ScalarExpression::Nil => todo!(),
            ScalarExpression::Cons(car, cdr) => {
                vec![*car.tag(), *car.value(), *cdr.tag(), *cdr.value()]
            }
            ScalarExpression::Comm(a, b) => {
                vec![a, *b.tag(), *b.value()]
            }
            ScalarExpression::Sym(string) => {
                todo!()
            }
            ScalarExpression::Fun {
                arg,
                body,
                closed_env,
            } => {
                vec![
                    *arg.tag(),
                    *arg.value(),
                    *body.tag(),
                    *body.value(),
                    *closed_env.tag(),
                    *closed_env.value(),
                ]
            }
            ScalarExpression::Str(string) => {
                todo!()
            }
            ScalarExpression::Thunk(thunk) => {
                vec![
                    *thunk.value.tag(),
                    *thunk.value.value(),
                    *thunk.continuation.tag(),
                    *thunk.continuation.value(),
                ]
            }
            ScalarExpression::Char(c) => {
                vec![char_to_f(c).unwrap()]
            }
            ScalarExpression::Num(x) => {
                vec![x]
            }
        }
    }

    pub fn de_f(stream: &Vec<F>, idx: usize, tag: Tag) -> Result<(usize, ScalarExpression<F>), ()> {
        match tag {
            Tag::Nil => todo!(),
            Tag::Cons => {
                let car = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let cdr = ScalarPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                Ok((idx + 4, ScalarExpression::Cons(car, cdr)))
            }
            Tag::Sym => todo!(),
            Tag::Comm => {
                let a = stream[idx];
                let b = ScalarPtr::from_parts(stream[idx + 1], stream[idx + 2]);
                Ok((idx + 3, ScalarExpression::Comm(a, b)))
            }
            Tag::Fun => {
                let arg = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let body = ScalarPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                let closed_env = ScalarPtr::from_parts(stream[idx + 4], stream[idx + 5]);
                Ok((
                    idx + 6,
                    ScalarExpression::Fun {
                        arg,
                        body,
                        closed_env,
                    },
                ))
            }
            Tag::Str => todo!(),
            Tag::Thunk => {
                let value = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                Ok((
                    idx + 4,
                    ScalarExpression::Thunk(ScalarThunk {
                        value,
                        continuation,
                    }),
                ))
            }
            Tag::Char => {
                let chr = stream[idx].to_u32().map_or_else(|| Err(()), |x| Ok(x))?;
                let chr = char::from_u32(chr).map_or_else(|| Err(()), |x| Ok(x))?;
                Ok((idx + 1, ScalarExpression::Char(chr)))
            }
            Tag::Num => Ok((idx + 1, ScalarExpression::Num(stream[idx]))),

            _ => Err(()),
        }
    }
}

// Unused for now, but will be needed when we serialize Thunks to IPLD.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarThunk<F: LurkField> {
    pub(crate) value: ScalarPtr<F>,
    pub(crate) continuation: ScalarContPtr<F>,
}

impl<F: LurkField> Copy for ScalarThunk<F> {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarContinuation<F: LurkField> {
    Outermost,
    Call {
        unevaled_arg: ScalarPtr<F>,
        saved_env: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Call2 {
        function: ScalarPtr<F>,
        saved_env: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Tail {
        saved_env: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Error,
    Lookup {
        saved_env: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Unop {
        operator: Op1,
        continuation: ScalarContPtr<F>,
    },
    Binop {
        operator: Op2,
        saved_env: ScalarPtr<F>,
        unevaled_args: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Binop2 {
        operator: Op2,
        evaled_arg: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Relop {
        operator: Rel2,
        saved_env: ScalarPtr<F>,
        unevaled_args: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Relop2 {
        operator: Rel2,
        evaled_arg: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    If {
        unevaled_args: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Let {
        var: ScalarPtr<F>,
        body: ScalarPtr<F>,
        saved_env: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    LetRec {
        var: ScalarPtr<F>,
        body: ScalarPtr<F>,
        saved_env: ScalarPtr<F>,
        continuation: ScalarContPtr<F>,
    },
    Emit {
        continuation: ScalarContPtr<F>,
    },
    Dummy,
    Terminal,
}

impl<F: LurkField> ScalarContinuation<F> {
    pub fn ser_f(self) -> Vec<F> {
        match self {
            ScalarContinuation::Outermost => todo!(),
            ScalarContinuation::Call {
                unevaled_arg,
                saved_env,
                continuation,
            } => {
                vec![
                    *unevaled_arg.tag(),
                    *unevaled_arg.value(),
                    *saved_env.tag(),
                    *saved_env.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Call2 {
                function,
                saved_env,
                continuation,
            } => {
                vec![
                    *function.tag(),
                    *function.value(),
                    *saved_env.tag(),
                    *saved_env.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Tail {
                saved_env,
                continuation,
            } => {
                vec![
                    *saved_env.tag(),
                    *saved_env.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Error => todo!(),
            ScalarContinuation::Lookup {
                saved_env,
                continuation,
            } => {
                vec![
                    *saved_env.tag(),
                    *saved_env.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Unop {
                operator,
                continuation,
            } => {
                vec![
                    operator.as_field(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Binop {
                operator,
                saved_env,
                unevaled_args,
                continuation,
            } => {
                vec![
                    operator.as_field(),
                    *saved_env.tag(),
                    *saved_env.value(),
                    *unevaled_args.tag(),
                    *unevaled_args.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Binop2 {
                operator,
                evaled_arg,
                continuation,
            } => {
                vec![
                    operator.as_field(),
                    *evaled_arg.tag(),
                    *evaled_arg.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Relop {
                operator,
                saved_env,
                unevaled_args,
                continuation,
            } => {
                vec![
                    operator.as_field(),
                    *saved_env.tag(),
                    *saved_env.value(),
                    *unevaled_args.tag(),
                    *unevaled_args.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Relop2 {
                operator,
                evaled_arg,
                continuation,
            } => {
                vec![
                    operator.as_field(),
                    *evaled_arg.tag(),
                    *evaled_arg.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::If {
                unevaled_args,
                continuation,
            } => {
                vec![
                    *unevaled_args.tag(),
                    *unevaled_args.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Let {
                var,
                body,
                saved_env,
                continuation,
            } => {
                vec![
                    *var.tag(),
                    *var.value(),
                    *body.tag(),
                    *body.value(),
                    *saved_env.tag(),
                    *saved_env.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::LetRec {
                var,
                body,
                saved_env,
                continuation,
            } => {
                vec![
                    *var.tag(),
                    *var.value(),
                    *body.tag(),
                    *body.value(),
                    *saved_env.tag(),
                    *saved_env.value(),
                    *continuation.tag(),
                    *continuation.value(),
                ]
            }
            ScalarContinuation::Emit { continuation } => {
                vec![*continuation.tag(), *continuation.value()]
            }
            ScalarContinuation::Dummy => todo!(),
            ScalarContinuation::Terminal => todo!(),
        }
    }
    pub fn de_f(
        stream: &Vec<F>,
        idx: usize,
        tag: ContTag,
    ) -> Result<(usize, ScalarContinuation<F>), ()> {
        match tag {
            ContTag::Outermost => todo!(),
            ContTag::Call0 => todo!(),
            ContTag::Call => {
                let unevaled_arg = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let saved_env = ScalarPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 4], stream[idx + 5]);
                Ok((
                    idx + 6,
                    ScalarContinuation::Call {
                        unevaled_arg,
                        saved_env,
                        continuation,
                    },
                ))
            }
            ContTag::Call2 => {
                let function = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let saved_env = ScalarPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 4], stream[idx + 5]);
                Ok((
                    idx + 6,
                    ScalarContinuation::Call2 {
                        function,
                        saved_env,
                        continuation,
                    },
                ))
            }
            ContTag::Tail => {
                let saved_env = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                Ok((
                    idx + 4,
                    ScalarContinuation::Tail {
                        saved_env,
                        continuation,
                    },
                ))
            }
            ContTag::Lookup => {
                let saved_env = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                Ok((
                    idx + 4,
                    ScalarContinuation::Lookup {
                        saved_env,
                        continuation,
                    },
                ))
            }
            ContTag::Unop => {
                let operator = Op1::from_field(stream[idx]).map_or_else(|| Err(()), Ok)?;
                let continuation = ScalarContPtr::from_parts(stream[idx + 1], stream[idx + 2]);
                Ok((
                    idx + 3,
                    ScalarContinuation::Unop {
                        operator,
                        continuation,
                    },
                ))
            }
            ContTag::Binop => {
                let operator = Op2::from_field(stream[idx]).map_or_else(|| Err(()), Ok)?;
                let saved_env = ScalarPtr::from_parts(stream[idx + 1], stream[idx + 2]);
                let unevaled_args = ScalarPtr::from_parts(stream[idx + 3], stream[idx + 4]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 5], stream[idx + 6]);
                Ok((
                    idx + 7,
                    ScalarContinuation::Binop {
                        operator,
                        saved_env,
                        unevaled_args,
                        continuation,
                    },
                ))
            }
            ContTag::Binop2 => {
                let operator = Op2::from_field(stream[idx]).map_or_else(|| Err(()), Ok)?;
                let evaled_arg = ScalarPtr::from_parts(stream[idx + 1], stream[idx + 2]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 3], stream[idx + 4]);
                Ok((
                    idx + 5,
                    ScalarContinuation::Binop2 {
                        operator,
                        evaled_arg,
                        continuation,
                    },
                ))
            }
            ContTag::Relop => {
                let operator = Rel2::from_field(stream[idx]).map_or_else(|| Err(()), Ok)?;
                let saved_env = ScalarPtr::from_parts(stream[idx + 1], stream[idx + 2]);
                let unevaled_args = ScalarPtr::from_parts(stream[idx + 3], stream[idx + 4]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 5], stream[idx + 6]);
                Ok((
                    idx + 7,
                    ScalarContinuation::Relop {
                        operator,
                        saved_env,
                        unevaled_args,
                        continuation,
                    },
                ))
            }
            ContTag::Relop2 => {
                let operator = Rel2::from_field(stream[idx]).map_or_else(|| Err(()), Ok)?;
                let evaled_arg = ScalarPtr::from_parts(stream[idx + 1], stream[idx + 2]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 3], stream[idx + 4]);
                Ok((
                    idx + 5,
                    ScalarContinuation::Relop2 {
                        operator,
                        evaled_arg,
                        continuation,
                    },
                ))
            }
            ContTag::If => {
                let unevaled_args = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                Ok((
                    idx + 4,
                    ScalarContinuation::If {
                        unevaled_args,
                        continuation,
                    },
                ))
            }
            ContTag::Let => {
                let var = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let body = ScalarPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                let saved_env = ScalarPtr::from_parts(stream[idx + 4], stream[idx + 5]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 6], stream[idx + 7]);
                Ok((
                    idx + 8,
                    ScalarContinuation::Let {
                        var,
                        body,
                        saved_env,
                        continuation,
                    },
                ))
            }
            ContTag::LetRec => {
                let var = ScalarPtr::from_parts(stream[idx], stream[idx + 1]);
                let body = ScalarPtr::from_parts(stream[idx + 2], stream[idx + 3]);
                let saved_env = ScalarPtr::from_parts(stream[idx + 4], stream[idx + 5]);
                let continuation = ScalarContPtr::from_parts(stream[idx + 6], stream[idx + 7]);
                Ok((
                    idx + 8,
                    ScalarContinuation::LetRec {
                        var,
                        body,
                        saved_env,
                        continuation,
                    },
                ))
            }
            ContTag::Emit => {
                let continuation = ScalarContPtr::from_parts(stream[idx], stream[idx + 1]);
                Ok((idx + 1, ScalarContinuation::Emit { continuation }))
            }
            ContTag::Dummy => todo!(),
            ContTag::Terminal => todo!(),

            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::eval::empty_sym_env;
    use crate::field::FWrap;
    use crate::store::ScalarPointer;
    use blstrs::Scalar as Fr;

    use quickcheck::{Arbitrary, Gen};

    use crate::test::frequency;

    use libipld::serde::from_ipld;
    use libipld::serde::to_ipld;

    impl Arbitrary for ScalarThunk<Fr> {
        fn arbitrary(g: &mut Gen) -> Self {
            ScalarThunk {
                value: Arbitrary::arbitrary(g),
                continuation: Arbitrary::arbitrary(g),
            }
        }
    }

    #[quickcheck]
    fn prop_scalar_thunk_ipld(x: ScalarThunk<Fr>) -> bool {
        if let Ok(ipld) = to_ipld(x) {
            if let Ok(y) = from_ipld(ipld) {
                x == y
            } else {
                false
            }
        } else {
            false
        }
    }

    impl Arbitrary for ScalarExpression<Fr> {
        fn arbitrary(g: &mut Gen) -> Self {
            let input: Vec<(i64, Box<dyn Fn(&mut Gen) -> ScalarExpression<Fr>>)> = vec![
                (100, Box::new(|_| Self::Nil)),
                (
                    100,
                    Box::new(|g| Self::Cons(ScalarPtr::arbitrary(g), ScalarPtr::arbitrary(g))),
                ),
                (100, Box::new(|g| Self::Sym(String::arbitrary(g)))),
                (100, Box::new(|g| Self::Str(String::arbitrary(g)))),
                (
                    100,
                    Box::new(|g| {
                        let f = FWrap::arbitrary(g);
                        Self::Num(f.0)
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Fun {
                        arg: ScalarPtr::arbitrary(g),
                        body: ScalarPtr::arbitrary(g),
                        closed_env: ScalarPtr::arbitrary(g),
                    }),
                ),
                (100, Box::new(|g| Self::Thunk(ScalarThunk::arbitrary(g)))),
            ];
            frequency(g, input)
        }
    }

    impl Arbitrary for ScalarContinuation<Fr> {
        fn arbitrary(g: &mut Gen) -> Self {
            let input: Vec<(i64, Box<dyn Fn(&mut Gen) -> ScalarContinuation<Fr>>)> = vec![
                (100, Box::new(|_| Self::Outermost)),
                (
                    100,
                    Box::new(|g| Self::Call {
                        unevaled_arg: ScalarPtr::arbitrary(g),
                        saved_env: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Call2 {
                        function: ScalarPtr::arbitrary(g),
                        saved_env: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Tail {
                        saved_env: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (100, Box::new(|_| Self::Error)),
                (
                    100,
                    Box::new(|g| Self::Lookup {
                        saved_env: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Unop {
                        operator: Op1::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Binop {
                        operator: Op2::arbitrary(g),
                        saved_env: ScalarPtr::arbitrary(g),
                        unevaled_args: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Binop2 {
                        operator: Op2::arbitrary(g),
                        evaled_arg: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Relop {
                        operator: Rel2::arbitrary(g),
                        saved_env: ScalarPtr::arbitrary(g),
                        unevaled_args: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Relop2 {
                        operator: Rel2::arbitrary(g),
                        evaled_arg: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::If {
                        unevaled_args: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::Let {
                        var: ScalarPtr::arbitrary(g),
                        body: ScalarPtr::arbitrary(g),
                        saved_env: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (
                    100,
                    Box::new(|g| Self::LetRec {
                        var: ScalarPtr::arbitrary(g),
                        saved_env: ScalarPtr::arbitrary(g),
                        body: ScalarPtr::arbitrary(g),
                        continuation: ScalarContPtr::arbitrary(g),
                    }),
                ),
                (100, Box::new(|_| Self::Dummy)),
                (100, Box::new(|_| Self::Terminal)),
            ];
            frequency(g, input)
        }
    }

    #[quickcheck]
    fn prop_scalar_expression_ipld(x: ScalarExpression<Fr>) -> bool {
        match to_ipld(x.clone()) {
            Ok(ipld) => match from_ipld(ipld.clone()) {
                Ok(y) => {
                    println!("x: {:?}", x);
                    println!("y: {:?}", y);
                    x == y
                }
                Err(e) => {
                    println!("ser x: {:?}", x);
                    println!("de ipld: {:?}", ipld);
                    println!("err e: {:?}", e);
                    false
                }
            },
            Err(e) => {
                println!("ser x: {:?}", x);
                println!("err e: {:?}", e);
                false
            }
        }
    }

    #[quickcheck]
    fn prop_scalar_continuation_ipld(x: ScalarExpression<Fr>) -> bool {
        if let Ok(ipld) = to_ipld(x.clone()) {
            if let Ok(y) = from_ipld(ipld) {
                x == y
            } else {
                false
            }
        } else {
            false
        }
    }

    // This doesn't create well-defined ScalarStores, so is only useful for
    // testing ipld
    impl Arbitrary for ScalarStore<Fr> {
        fn arbitrary(g: &mut Gen) -> Self {
            let map: Vec<(ScalarPtr<Fr>, Option<ScalarExpression<Fr>>)> = Arbitrary::arbitrary(g);
            let cont_map: Vec<(ScalarContPtr<Fr>, Option<ScalarContinuation<Fr>>)> =
                Arbitrary::arbitrary(g);
            ScalarStore {
                scalar_map: map.into_iter().collect(),
                scalar_cont_map: cont_map.into_iter().collect(),
            }
        }
    }

    #[quickcheck]
    fn prop_scalar_store_ipld(x: ScalarStore<Fr>) -> bool {
        if let Ok(ipld) = to_ipld(x.clone()) {
            if let Ok(y) = from_ipld(ipld) {
                x == y
            } else {
                false
            }
        } else {
            false
        }
    }

    #[test]
    fn test_expr_ipld() {
        let test = |src| {
            let mut store1 = Store::<Fr>::default();
            let expr1 = store1.read(src).unwrap();
            store1.hydrate_scalar_cache();

            if let (scalar_store, Some(scalar_expr)) = ScalarStore::new_with_expr(&store1, &expr1) {
                let ipld = to_ipld(scalar_store.clone()).unwrap();
                let mut scalar_store2 = from_ipld(ipld).unwrap();
                assert_eq!(scalar_store, scalar_store2);
                if let Some((mut store2, expr2)) = scalar_store2.to_store_with_expr(&scalar_expr) {
                    store2.hydrate_scalar_cache();
                    let (scalar_store3, _) = ScalarStore::new_with_expr(&store2, &expr2);
                    assert_eq!(scalar_store2, scalar_store3)
                } else {
                    assert!(false)
                }
            } else {
                assert!(false)
            }
        };

        test("symbol");
        test("1");
        test("(1 . 2)");
        test("\"foo\" . \"bar\")");
        test("(foo . bar)");
        test("(1 . 2)");
        test("(+ 1 2 3)");
        test("(+ 1 2 (* 3 4))");
        test("(+ 1 2 (* 3 4) \"asdf\" )");
        test("(+ 1 2 2 (* 3 4) \"asdf\" \"asdf\")");
    }

    #[test]
    fn test_expr_eval_ipld() {
        use crate::eval;
        let test = |src| {
            let mut s = Store::<Fr>::default();
            let expr = s.read(src).unwrap();
            s.hydrate_scalar_cache();

            let env = empty_sym_env(&s);
            let mut eval = eval::Evaluator::new(expr, env, &mut s, 100);
            let (
                eval::IO {
                    expr,
                    env: _,
                    cont: _,
                },
                _lim,
                _emitted,
            ) = eval.eval();

            let (scalar_store, _) = ScalarStore::new_with_expr(&s, &expr);
            println!("{:?}", scalar_store);
            let ipld = to_ipld(scalar_store.clone()).unwrap();
            let scalar_store2 = from_ipld(ipld).unwrap();
            println!("{:?}", scalar_store2);
            assert_eq!(scalar_store, scalar_store2);
        };

        test("(let ((a 123)) (lambda (x) (+ x a)))");
    }

    #[test]
    fn test_scalar_store() {
        let test = |src, expected| {
            let mut s = Store::<Fr>::default();
            let expr = s.read(src).unwrap();
            s.hydrate_scalar_cache();

            let (scalar_store, _) = ScalarStore::new_with_expr(&s, &expr);
            assert_eq!(expected, scalar_store.scalar_map.len());
        };

        // Four atoms, four conses (four-element list), and NIL.
        test("symbol", 1);
        test("(1 . 2)", 3);
        test("(+ 1 2 3)", 9);
        test("(+ 1 2 (* 3 4))", 14);
        // String are handled.
        test("(+ 1 2 (* 3 4) \"asdf\" )", 16);
        // Duplicate strings or symbols appear only once.
        test("(+ 1 2 2 (* 3 4) \"asdf\" \"asdf\")", 18);
    }

    #[test]
    fn test_scalar_store_opaque_cons() {
        let mut store = Store::<Fr>::default();

        let num1 = store.num(123);
        let num2 = store.num(987);
        let cons = store.intern_cons(num1, num2);
        let cons_hash = store.hash_expr(&cons).unwrap();
        let opaque_cons = store.intern_maybe_opaque_cons(*cons_hash.value());

        store.hydrate_scalar_cache();

        let (scalar_store, _) = ScalarStore::new_with_expr(&store, &opaque_cons);
        println!("{:?}", scalar_store.scalar_map);
        println!("{:?}", scalar_store.scalar_map.len());
        // If a non-opaque version has been found when interning opaque, children appear in `ScalarStore`.
        assert_eq!(3, scalar_store.scalar_map.len());
    }
    #[test]
    fn test_scalar_store_opaque_fun() {
        let mut store = Store::<Fr>::default();

        let arg = store.sym("A");
        let body = store.num(123);
        let empty_env = empty_sym_env(&store);
        let fun = store.intern_fun(arg, body, empty_env);
        let fun_hash = store.hash_expr(&fun).unwrap();
        let opaque_fun = store.intern_maybe_opaque_fun(*fun_hash.value());
        store.hydrate_scalar_cache();

        let (scalar_store, _) = ScalarStore::new_with_expr(&store, &opaque_fun);
        // If a non-opaque version has been found when interning opaque, children appear in `ScalarStore`.
        assert_eq!(4, scalar_store.scalar_map.len());
    }
    #[test]
    fn test_scalar_store_opaque_sym() {
        let mut store = Store::<Fr>::default();

        let sym = store.sym(&"sym");
        let sym_hash = store.hash_expr(&sym).unwrap();
        let opaque_sym = store.intern_maybe_opaque_sym(*sym_hash.value());

        store.hydrate_scalar_cache();

        let (scalar_store, _) = ScalarStore::new_with_expr(&store, &opaque_sym);
        // Only the symbol's string itself, not all of its substrings, appears in `ScalarStore`.
        assert_eq!(1, scalar_store.scalar_map.len());
    }
    #[test]
    fn test_scalar_store_opaque_str() {
        let mut store = Store::<Fr>::default();

        let str = store.str(&"str");
        let str_hash = store.hash_expr(&str).unwrap();
        let opaque_str = store.intern_maybe_opaque_sym(*str_hash.value());

        store.hydrate_scalar_cache();

        let (scalar_store, _) = ScalarStore::new_with_expr(&store, &opaque_str);
        // Only the string itself, not all of its substrings, appears in `ScalarStore`.
        assert_eq!(1, scalar_store.scalar_map.len());
    }
    #[test]
    fn test_scalar_store_opaque_comm() {
        let mut store = Store::<Fr>::default();

        let num = store.num(987);
        let comm = store.intern_comm(Fr::from(123), num);
        let comm_hash = store.hash_expr(&comm).unwrap();
        let opaque_comm = store.intern_maybe_opaque_comm(*comm_hash.value());

        store.hydrate_scalar_cache();

        let (scalar_store, _) = ScalarStore::new_with_expr(&store, &opaque_comm);
        println!("{:?}", scalar_store.scalar_map);
        println!("{:?}", scalar_store.scalar_map.len());
        // If a non-opaque version has been found when interning opaque, children appear in `ScalarStore`.
        assert_eq!(2, scalar_store.scalar_map.len());
    }
}
