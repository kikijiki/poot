//! Fixture for the model-free lane's self-test (`scripts/test-model-free.sh self-test`): the nextest `gpu`
//! group must select a poot-llm test by its TEST name. This binary's name contains none of the group's words,
//! so only the test filter can place `a_hardware_named_test_joins_the_group` in the group, and the control
//! test, named without one, must stay out of it. The tests themselves do nothing.

#[test]
fn a_hardware_named_test_joins_the_group() {}

#[test]
fn a_host_only_test_stays_out_of_the_group() {}
