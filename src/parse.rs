pub type ParseResult<'a, Output> = Result<(Output, &'a str), &'a str>;

pub type BoxedParser<'a, Output> = Box<dyn Parser<'a, Output> + 'a>;

pub trait Parser<'a, Output> {
    fn pars(&self, input: &'a str) -> ParseResult<'a, Output>;

    fn parse_exact(&self, input: &'a str) -> Result<Output, &'a str> {
        self.pars(input).map(|pair| pair.0)
    }

    fn map<U, F>(self, map_fn: F) -> BoxedParser<'a, U>
    where
        Self: Sized + 'a,
        Output: 'a,
        U: 'a,
        F: Fn(Output) -> U + 'a,
    {
        Box::new(move |input: &'a str| {
            self.pars(input)
                .map(|(result, rest)| (map_fn(result), rest))
        })
    }

    fn yielding<T>(self, value: T) -> BoxedParser<'a, T>
    where
        Self: Sized + 'a,
        Output: 'a,
        T: Clone + 'a,
    {
        self.map(move |_| value.clone())
    }

    fn pair<Other, OtherOutput>(self, other: Other) -> BoxedParser<'a, (Output, OtherOutput)>
    where
        Self: Sized + 'a,
        Other: Parser<'a, OtherOutput> + 'a,
    {
        Box::new(move |input: &'a str| {
            self.pars(input).and_then(|(result, rest)| {
                other
                    .pars(rest)
                    .map(move |(other_result, rest)| ((result, other_result), rest))
            })
        })
    }

    fn or<Other>(self, other: Other) -> BoxedParser<'a, Output>
    where
        Self: Sized + 'a,
        Other: Parser<'a, Output> + 'a,
    {
        Box::new(move |input: &'a str| self.pars(input).or_else(|_| other.pars(input)))
    }

    fn left<Other, OtherOutput>(self, other: Other) -> BoxedParser<'a, Output>
    where
        Self: Sized + 'a,
        Other: Parser<'a, OtherOutput> + 'a,
    {
        Box::new(move |input: &'a str| {
            self.pars(input)
                .and_then(|(result, rest)| other.pars(rest).map(move |(_, rest)| (result, rest)))
        })
    }

    fn right<Other, OtherOutput>(self, other: Other) -> BoxedParser<'a, OtherOutput>
    where
        Self: Sized + 'a,
        Other: Parser<'a, OtherOutput> + 'a,
    {
        Box::new(move |input: &'a str| {
            self.pars(input)
                .and_then(|(_, rest)| other.pars(rest).map(move |(result, rest)| (result, rest)))
        })
    }

    fn interspersed<Other, OtherOutput>(self, separator: Other) -> BoxedParser<'a, Vec<Output>>
    where
        Self: Sized + 'a,
        Other: Parser<'a, OtherOutput> + 'a,
    {
        Box::new(move |input: &'a str| {
            if let Ok((result, mut rest)) = self.pars(input) {
                let mut results = vec![result];
                while let Ok((_, rem)) = separator.pars(rest) {
                    match self.pars(rem) {
                        Ok((result, rem)) => {
                            results.push(result);
                            rest = rem;
                        }
                        Err(_) => return Err(input),
                    }
                }
                Ok((results, rest))
            } else {
                Ok((Vec::new(), input))
            }
        })
    }
}

impl<'a, Output> Parser<'a, Output> for BoxedParser<'a, Output> {
    fn pars(&self, input: &'a str) -> ParseResult<'a, Output> {
        self.as_ref().pars(input)
    }
}

impl<'a> Parser<'a, char> for char {
    fn pars(&self, input: &'a str) -> ParseResult<'a, char> {
        let mut chars = input.chars();
        match chars.next() {
            Some(c) if &c == self => Ok((c, chars.as_str())),
            _ => Err(input),
        }
    }
}

impl<'a> Parser<'a, &'a str> for &'a str {
    fn pars(&self, input: &'a str) -> ParseResult<'a, &'a str> {
        input
            .strip_prefix(self)
            .map(|rest| (*self, rest))
            .ok_or(input)
    }
}

impl<'a, F, Output> Parser<'a, Output> for F
where
    F: Fn(&'a str) -> ParseResult<'a, Output>,
{
    fn pars(&self, input: &'a str) -> ParseResult<'a, Output> {
        self(input)
    }
}

pub fn any_char(input: &str) -> ParseResult<char> {
    let mut chars = input.chars();
    let c = chars.next().ok_or(input)?;
    Ok((c, chars.as_str()))
}

pub fn predicate<'a, F>(pred_fn: F) -> impl Parser<'a, &'a str>
where
    F: Fn(char) -> bool + Clone,
{
    move |input: &'a str| {
        let rest = input.trim_start_matches(pred_fn.clone());
        if rest.len() == input.len() {
            Err(input)
        } else {
            Ok((&input[..(input.len() - rest.len())], rest))
        }
    }
}

pub fn opt<'a, P, Output>(parser: P) -> impl Parser<'a, Option<Output>>
where
    P: Parser<'a, Output>,
{
    move |input: &'a str| match parser.pars(input) {
        Ok((item, rest)) => Ok((Some(item), rest)),
        Err(_) => Ok((None, input)),
    }
}

pub fn zero_or_more<'a, P, Output>(parser: P) -> impl Parser<'a, Vec<Output>>
where
    P: Parser<'a, Output>,
{
    move |mut input: &'a str| {
        let mut results = Vec::new();
        while let Ok((result, rest)) = parser.pars(input) {
            results.push(result);
            input = rest;
        }
        Ok((results, input))
    }
}

pub fn whitespace(input: &str) -> ParseResult<()> {
    predicate(char::is_whitespace).map(|_| ()).pars(input)
}

pub fn word(input: &str) -> ParseResult<&str> {
    predicate(char::is_alphabetic).pars(input)
}

pub fn trimmed<'a, P, Output>(parser: P) -> impl Parser<'a, Output>
where
    P: Parser<'a, Output> + 'a,
    Output: 'a,
{
    opt(whitespace).right(parser).left(opt(whitespace))
}

pub fn uint<'a, T: std::str::FromStr<Err = impl std::fmt::Debug> + 'a>(
    input: &'a str,
) -> ParseResult<T> {
    predicate(char::is_numeric)
        .map(|s| s.parse::<T>().unwrap())
        .pars(input)
}

pub fn int<
    'a,
    T: std::str::FromStr<Err = impl std::fmt::Debug> + std::ops::Neg<Output = T> + 'a,
>(
    input: &'a str,
) -> ParseResult<T> {
    opt('-')
        .pair(predicate(char::is_numeric))
        .map(|(sign, s)| {
            let num = s.parse::<T>().unwrap();
            if let Some('-') = sign {
                -num
            } else {
                num
            }
        })
        .pars(input)
}

#[test]
fn test() {
    assert_eq!("hello".pars("hello, world!"), Ok(("hello", ", world!")));
    let parse_123 = "123".map(|s| s.parse::<usize>().unwrap());
    assert_eq!(parse_123.pars("12345"), Ok((123, "45")));
    assert_eq!('-'.pair(parse_123).pars("-12345"), Ok((('-', 123), "45")));

    assert_eq!(uint.pars("123a"), Ok((123, "a")));

    let comma_sep = uint.interspersed(',');

    assert_eq!(
        comma_sep.pars("1,2,3,4nonnum,134"),
        Ok((vec![1, 2, 3, 4], "nonnum,134"))
    );

    assert_eq!(
        '['.right(comma_sep).left(']').pars("[1,2,3,4]nonnum,134]"),
        Ok((vec![1, 2, 3, 4], "nonnum,134]"))
    );

    assert_eq!(int.pars("-12345"), Ok((-12345, "")));
    assert_eq!(int.pars("-345123"), Ok((-345123, "")));
    assert_eq!(int.pars("100000"), Ok((100000, "")));

    assert_eq!(
        whitespace.right(int).left(whitespace).pars("   -100 \t\n"),
        Ok((-100, ""))
    );
}
