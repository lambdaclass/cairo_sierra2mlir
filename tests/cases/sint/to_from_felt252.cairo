use traits::TryInto;

fn main() -> (
    felt252, felt252, Option<i8>, Option<i8>, Option<i8>,
    felt252, felt252, Option<i16>, Option<i16>, Option<i16>,
    felt252, felt252, Option<i32>, Option<i32>, Option<i32>,
) {
    (
        17_i8.into(),
        -17_i8.into(),
        17.try_into(),
        150.try_into(),
        24857893469346.try_into(),

        17_i16.into(),
        -17_i16.into(),
        17.try_into(),
        270.try_into(),
        24857893469346.try_into(),

        17_i16.into(),
        -17_i16.into(),
        17.try_into(),
        2147483649.try_into(),
        24857893469346.try_into(),
    )
}
