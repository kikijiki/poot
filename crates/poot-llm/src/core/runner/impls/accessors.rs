use super::super::Runner;

impl Runner {
    /// Which weight constants this checkpoint stores packed (card 545a): the record every traced graph
    /// is bound with ([`Runner::bind_storage`]). Empty for a dense checkpoint.
    pub fn weight_formats(&self) -> &poot_graph_plan::WeightFormats {
        &self.formats
    }
}
