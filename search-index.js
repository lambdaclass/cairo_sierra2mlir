var searchIndex = JSON.parse('{\
"mlir":{"doc":"A MLIR safe API wrapper","t":[0,0,0,0,0,0,3,11,11,11,11,11,11,11,11,11,11,3,11,11,11,11,11,11,11,11,11,11,11,11,11,3,3,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,3,11,11,11,11,11,11,11,11,11,4,13,13,3,13,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,3,11,11,11,11,11,11,11,11,11,11],"n":["block","context","dialects","location","mlir_type","module","Block","borrow","borrow_mut","drop","fmt","from","into","new","try_from","try_into","type_id","Context","append_registry","borrow","borrow_mut","default","drop","eq","fmt","from","into","new","try_from","try_into","type_id","Dialect","Registry","borrow","borrow","borrow_mut","borrow_mut","default","drop","fmt","from","from","into","into","new","try_from","try_from","try_into","try_into","type_id","type_id","Location","borrow","borrow_mut","fmt","from","into","new","try_from","try_into","type_id","DataType","Int","SignedInt","Type","UnsignedInt","borrow","borrow","borrow_mut","borrow_mut","clone","clone_into","eq","fmt","fmt","from","from","get_width","into","into","is_int","new","to_owned","try_from","try_from","try_into","try_into","type_id","type_id","bitwidth","bitwidth","bitwidth","Module","borrow","borrow_mut","drop","fmt","from","into","new","try_from","try_into","type_id"],"q":["mlir","","","","","","mlir::block","","","","","","","","","","","mlir::context","","","","","","","","","","","","","","mlir::dialects","","","","","","","","","","","","","","","","","","","","mlir::location","","","","","","","","","","mlir::mlir_type","","","","","","","","","","","","","","","","","","","","","","","","","","","","mlir::mlir_type::DataType","","","mlir::module","","","","","","","","","",""],"d":["","","","","","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Creates a new empty block with the given argument types …","","","","","Append the contents of the given dialect registry to the …","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Creates an MLIR context.","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","A MLIR location.","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Creates a location with unknown position owned by the …","","","","","","","A MLIR type.","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Gets the bit width of this type, if it is an integer type.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","","","",""],"i":[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,8,8,8,8,8,8,8,8,8,8,8,8,8,0,0,16,9,16,9,9,9,9,16,9,16,9,9,16,9,16,9,16,9,0,11,11,11,11,11,11,11,11,11,0,12,12,0,12,12,13,12,13,12,12,12,12,13,12,13,13,12,13,12,13,12,12,13,12,13,12,13,17,18,19,0,15,15,15,15,15,15,15,15,15,15],"f":[0,0,0,0,0,0,0,[[]],[[]],[1],[[1,2],3],[[]],[[]],[[[5,[4]]],1],[[],6],[[],6],[[],7],0,[[8,9]],[[]],[[]],[[],8],[8],[[8,8],10],[[8,2],3],[[]],[[]],[[],8],[[],6],[[],6],[[],7],0,0,[[]],[[]],[[]],[[]],[[],9],[9],[[9,2],3],[[]],[[]],[[]],[[]],[[],9],[[],6],[[],6],[[],6],[[],6],[[],7],[[],7],0,[[]],[[]],[[11,2],3],[[]],[[]],[8,11],[[],6],[[],6],[[],7],0,0,0,0,0,[[]],[[]],[[]],[[]],[12,12],[[]],[[12,12],10],[[12,2],3],[[13,2],3],[[]],[[]],[13,[[5,[14]]]],[[]],[[]],[12,10],[[8,12],13],[[]],[[],6],[[],6],[[],6],[[],6],[[],7],[[],7],0,0,0,0,[[]],[[]],[15],[[15,2],3],[[]],[[]],[11,15],[[],6],[[],6],[[],7]],"p":[[3,"Block"],[3,"Formatter"],[6,"Result"],[3,"Vec"],[4,"Option"],[4,"Result"],[3,"TypeId"],[3,"Context"],[3,"Registry"],[15,"bool"],[3,"Location"],[4,"DataType"],[3,"Type"],[15,"u32"],[3,"Module"],[3,"Dialect"],[13,"SignedInt"],[13,"Int"],[13,"UnsignedInt"]]},\
"sierra2mlir":{"doc":"A compiler to convert Cairo’s intermediate …","t":[3,11,11,11,11,11,11,11,11,11,11,11,12,11,5,12,11,11,11,11,11],"n":["Args","augment_args","augment_args_for_update","borrow","borrow_mut","command","command_for_update","fmt","from","from_arg_matches","from_arg_matches_mut","group_id","input","into","main","output","try_from","try_into","type_id","update_from_arg_matches","update_from_arg_matches_mut"],"q":["sierra2mlir","","","","","","","","","","","","","","","","","","","",""],"d":["","","","","","","","","Returns the argument unchanged.","","","","The input sierra file.","Calls <code>U::from(self)</code>.","","The output file.","","","","",""],"i":[0,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2,2],"f":[0,[1,1],[1,1],[[]],[[]],[[],1],[[],1],[[2,3],4],[[]],[5,[[7,[2,6]]]],[5,[[7,[2,6]]]],[[],[[9,[8]]]],0,[[]],[[]],0,[[],7],[[],7],[[],10],[[2,5],[[7,[6]]]],[[2,5],[[7,[6]]]]],"p":[[3,"Command"],[3,"Args"],[3,"Formatter"],[6,"Result"],[3,"ArgMatches"],[6,"Error"],[4,"Result"],[3,"Id"],[4,"Option"],[3,"TypeId"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};
