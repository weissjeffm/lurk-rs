use crate::eval::{empty_sym_env, Evaluator, IO};
use crate::field::LurkField;
use crate::store::{ContPtr, ContTag, Expression, Pointer, Ptr, Store, Tag};
use crate::writer::Write;
use anyhow::Result;
use rustyline::error::ReadlineError;
use rustyline::validate::{
    MatchingBracketValidator, ValidationContext, ValidationResult, Validator,
};
use rustyline::{Config, Editor};
use rustyline_derive::{Completer, Helper, Highlighter, Hinter};
use std::fs::read_to_string;
use std::io::{self, StdoutLock, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Completer, Helper, Highlighter, Hinter)]
pub struct InputValidator {
    pub brackets: MatchingBracketValidator,
}

impl Validator for InputValidator {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        self.brackets.validate(ctx)
    }
}
pub enum ReplError {
    Interrupted,
    Eof,
    Other(String),
}

#[derive(Clone)]
pub struct ReplState<F: LurkField> {
    env: Ptr<F>,
    limit: usize,
}

pub struct Repl<F: LurkField> {
    state: ReplState<F>,
    rl: Editor<InputValidator>,
    history_path: PathBuf,
    stdout: StdoutLock<'static>,
}

impl<F: LurkField> Repl<F> {
    pub fn new(s: &mut Store<F>, limit: usize) -> Result<Self> {
        let history_path = dirs::home_dir()
            .expect("missing home directory")
            .join(".lurk-history");

        let h = InputValidator {
            brackets: MatchingBracketValidator::new(),
        };
        let config = Config::builder()
            .color_mode(rustyline::ColorMode::Enabled)
            .auto_add_history(true)
            .build();
        let mut rl = Editor::with_config(config);
        rl.set_helper(Some(h));

        let stdout = io::stdout().lock();
        let state = Arc::new(Mutex::new(ReplState::new(store, limit)));
        Ok(Self {
            state,
            rl,
            history_path,
            stdout,
        })
    }
}

impl Repl for CliRepl {
    fn println(&self, s: String) -> Result<()> {
        println!("{}", s);
        Ok(())
    }

    fn writer<'a>(&'a mut self) -> &'a mut (dyn io::Write + 'a) {
        &mut self.stdout
    }

    fn save_history(&mut self) -> Result<()> {
        self.rl.save_history(&self.history_path)?;
        Ok(())
    }

    fn add_history_entry(&mut self, _s: &str) -> Result<()> {
        Ok(())
    }

    fn get_state(&self) -> Arc<Mutex<ReplState>> {
        self.state.clone()
    }

    fn load_history(&mut self) -> Result<()> {
        if self.history_path.exists() {
            self.rl.load_history(&self.history_path)?;
        }

        Ok(())
    }
}

// For the moment, input must be on a single line.
pub fn repl<P: AsRef<Path>, F: LurkField>(lurk_file: Option<P>) -> Result<()> {
    println!("Lurk REPL welcomes you.");

    let mut s = Store::<F>::default();
    let limit = 100_000_000;
    let mut repl = CliRepl::new(s, limit)?;
    repl.println("Lurk REPL welcomes you.".to_owned())?;

    {
        if let Some(lurk_file) = lurk_file {
            repl.state
                .lock()
                .map_err(|e| anyhow!("{}", e))?
                .handle_run(&lurk_file, &|s| println!("{}", s))
                .unwrap();
            return Ok(());
        }
    }

    loop {
        match repl.rl.readline("> ") {
            Ok(line) => {
                if let Ok(LineResult::Quit) = repl.handle_line(line) {
                    break;
                }
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                println!("Exiting...");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}

impl<F: LurkField> ReplState<F> {
    pub fn new(s: &mut Store<F>, limit: usize) -> Self {
        Self {
            store: store_mutex.clone(),
            env: empty_sym_env(&mut store_mutex.lock().unwrap()),
            limit,
        }
    }

    pub fn get_store(&self) -> Arc<Mutex<Store<Fr>>> {
        self.store.clone()
    }

    pub fn eval_expr(
        &mut self,
        expr: Ptr<F>,
        store: &mut Store<F>,
    ) -> (Ptr<F>, usize, ContPtr<F>, Vec<Ptr<F>>) {
        let (
            IO {
                expr: result,
                env: _env,
                cont: next_cont,
            },
            limit,
            emitted,
        ) = Evaluator::new(expr, self.env, store, self.limit).eval();

        (result, limit, next_cont, emitted)
    }

    /// Returns two bools.
    /// First bool is true if input is a command.
    /// Second bool is true if processing should continue.
    pub fn maybe_handle_command(
        &mut self,
        store: &mut Store<F>,
        line: &str,
        println: &dyn Fn(String),
    ) -> Result<(bool, bool)> {
        let store_mutex = self.store.clone();
        let mut store = store_mutex.lock().map_err(|e| anyhow!("{}", e))?;
        let mut chars = line.chars().peekable();
        let maybe_command = store.read_next(&mut chars);

        let result = match &maybe_command {
            Some(maybe_command) => match maybe_command.tag() {
                Tag::Sym => match store.fetch(maybe_command).unwrap().as_sym_str().unwrap() {
                    ":QUIT" => (true, false),
                    ":LOAD" => match store.read_string(&mut chars) {
                        Some(s) => match s.tag() {
                            Tag::Str => {
                                let path = store.fetch(&s).unwrap();
                                let path = PathBuf::from(path.as_str().unwrap());
                                self.handle_load(path, println)?;
                                (true, true)
                            }
                            other => {
                                anyhow::bail!("No valid path found: {:?}", other);
                            }
                        },
                        None => {
                            anyhow::bail!("No path found");
                        }
                    },
                    ":RUN" => {
                        if let Some(s) = store.read_string(&mut chars) {
                            if s.tag() == Tag::Str {
                                let path = store.fetch(&s).unwrap();
                                let path = PathBuf::from(path.as_str().unwrap());
                                self.handle_run(&path, println)?;
                            }
                        }
                        (true, true)
                    }
                    ":CLEAR" => {
                        self.env = empty_sym_env(&store);
                        (true, true)
                    }
                    s => {
                        if s.starts_with(':') {
                            println(format!("Unkown command: {}", s));
                            (true, true)
                        } else {
                            (false, true)
                        }
                    }
                },
                _ => (false, true),
            },
            _ => (false, true),
        };

        Ok(result)
    }

    pub fn handle_load<P: AsRef<Path>>(&mut self, store: &mut Store<F>, path: P) -> Result<()> {
        println!("Loading from {}.", path.as_ref().to_str().unwrap());
        let input = read_to_string(path)?;

        let store_mutex = self.store.clone();
        let mut store = store_mutex.lock().unwrap();
        let expr = store.read(&input).unwrap();
        let (result, _limit, _next_cont) = self.eval_expr(expr, &mut store);

        self.env = result;

        println(format!("Read: {}", input));
        io::stdout().flush().unwrap();
        Ok(())
    }

    pub fn handle_run<P: AsRef<Path> + Copy>(
        &mut self,
        store: &mut Store<F>,
        path: P,
        println: &dyn Fn(String),
    ) -> Result<()> {
        let store_mutex = self.store.clone();
        let mut store = store_mutex.lock().unwrap();
        println(format!("Running from {}.", path.as_ref().to_str().unwrap()));
        let p = path;

        let input = read_to_string(path)?;
        println(format!(
            "Read from {}: {}",
            path.as_ref().to_str().unwrap(),
            input
        ));
        let mut chars = input.chars().peekable();

        while let Some((ptr, is_meta)) = store.read_maybe_meta(&mut chars) {
            let expr = store.fetch(&ptr).unwrap();
            if is_meta {
                match expr {
                    Expression::Cons(car, rest) => match &store.fetch(&car).unwrap() {
                        Expression::Sym(s) => {
                            if s == &":LOAD" {
                                match store.fetch(&store.car(&rest)).unwrap() {
                                    Expression::Str(path) => {
                                        let joined =
                                            p.as_ref().parent().unwrap().join(Path::new(&path));
                                        self.handle_load(&joined, println)?
                                    }
                                    _ => panic!("Argument to :LOAD must be a string."),
                                }
                                io::stderr().flush().unwrap();
                            } else if s == &":RUN" {
                                match store.fetch(&store.car(&rest)).unwrap() {
                                    Expression::Str(path) => {
                                        let joined =
                                            p.as_ref().parent().unwrap().join(Path::new(&path));
                                        self.handle_run(&joined, println)?
                                    }
                                    _ => panic!("Argument to :RUN must be a string."),
                                }
                            } else if s == &":ASSERT-EQ" {
                                let (first, rest) = store.car_cdr(&rest);
                                let (second, rest) = store.car_cdr(&rest);
                                assert!(rest.is_nil());
                                let (first_evaled, _, _, _) = self.eval_expr(first, store);
                                let (second_evaled, _, _, _) = self.eval_expr(second, store);
                                assert!(store.ptr_eq(&first_evaled, &second_evaled));
                            } else if s == &":ASSERT" {
                                let (first, rest) = store.car_cdr(&rest);
                                assert!(rest.is_nil());
                                let (first_evaled, _, _) = self.eval_expr(first, &mut store);
                                assert!(!first_evaled.is_nil());
                            } else if s == &":CLEAR" {
                                self.env = empty_sym_env(&store);
                            } else if s == &":ASSERT-ERROR" {
                                let (first, rest) = store.car_cdr(&rest);

                                assert!(rest.is_nil());
                                let (_, _, continuation) =
                                    self.clone().eval_expr(first, &mut store);
                                assert!(continuation.is_error());
                                // FIXME: bring back catching, or solve otherwise
                                // std::panic::catch_unwind(||
                                // } else {
                                //     // There was a panic, so this is okay.
                                //     // FIXME: Never panic. Instead return Continuation::Error when evaluating.
                                //     ()
                                // }
                            } else if s == &":ASSERT-EMITTED" {
                                let (first, rest) = store.car_cdr(&rest);
                                let (second, rest) = store.car_cdr(&rest);

                                assert!(rest.is_nil());
                                let (first_evaled, _, _, _) = self.clone().eval_expr(first, store);
                                let (_, _, _, emitted) = self.eval_expr(second, store);
                                let (mut first_emitted, mut rest_emitted) =
                                    store.car_cdr(&first_evaled);
                                for (i, elem) in emitted.iter().enumerate() {
                                    if elem != &first_emitted {
                                        panic!(
                                            ":ASSERT-EMITTED failed at position {}. Expected {}, but found {}.",
                                            i,
                                            first_emitted.fmt_to_string(store),
                                            elem.fmt_to_string(store),
                                        );
                                    }
                                    (first_emitted, rest_emitted) = store.car_cdr(&rest_emitted);
                                }
                            } else {
                                panic!("!({} ...) is unsupported.", s);
                            }
                        }
                        _ => panic!("!(<COMMAND> ...) must be a (:keyword) symbol."),
                    },
                    _ => panic!("!<COMMAND> form is unsupported."),
                }
            } else {
                let (result, _limit, _next_cont) = self.eval_expr(ptr, &mut store);

                println(format!("Evaled: {}", result.fmt_to_string(&store)));
                io::stdout().flush().unwrap();
            }
        }

        Ok(())
    }
}
