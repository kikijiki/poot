//! Fixture for the model-free lane's self-test (`scripts/test-model-free.sh self-test`): the nextest `gpu`
//! group must select a poot-llm test by its BINARY name. This binary's name contains `hardware` and its test
//! name contains none of the group's words, so only the binary filter can place it in the group. The test
//! itself does nothing.

#[test]
fn only_the_binary_name_places_this_test_in_the_group() {}
