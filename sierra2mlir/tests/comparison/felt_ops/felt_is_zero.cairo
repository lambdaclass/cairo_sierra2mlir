fn main() -> (felt252, felt252, felt252, felt252, felt252, felt252, felt252, felt252) {
    (
        test(0),
        test(-0),
        test(1),
        test(10),
        test(3618502788666131213697322783095070105623107215331596699973092056135872020480),
        test(-1),
        test(-10),
        test(-3618502788666131213697322783095070105623107215331596699973092056135872020480),
    )
}

fn test(input: felt252) -> felt252 {
    if input == 0 {
        100
    } else {
        50
    }
}
