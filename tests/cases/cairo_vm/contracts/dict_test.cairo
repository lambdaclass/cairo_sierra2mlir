#[starknet::contract]
mod DictTest {
    use dict::Felt252DictTrait;
    use core::ops::index::Index;

    const KEY1: felt252 = 10;
    const KEY2: felt252 = 21;
    // KEY3 is ~37% * PRIME.
    const KEY3: felt252 =
        1343531647004637707094910297222796970954128321746173119103571679493202324940;
    // KEY4 and KEY5 are ~92% * PRIME.
    const KEY4: felt252 =
        3334603141101959564751596861783084684819726025596122159217101666076094555684;
    const KEY5: felt252 =
        3334603141101959564751596861783084684819726025596122159217101666076094555685;

    #[storage]
    struct Storage {}

    #[external(v0)]
    fn test_dict_big_keys(self: @ContractState) -> felt252 {
        let mut dict: Felt252Dict<felt252> = Default::default();

        dict.insert(KEY1, 1);
        dict.insert(KEY2, 2);
        dict.insert(KEY3, 3);
        dict.insert(KEY4, 4);
        dict.insert(KEY5, 5);

        assert(dict[KEY1] == 1, 'KEY1');
        assert(dict[KEY2] == 2, 'KEY2');
        assert(dict[KEY3] == 3, 'KEY3');
        assert(dict[KEY4] == 4, 'KEY4');
        assert(dict[KEY5] == 5, 'KEY5');

        return dict[KEY5];
    }
}
