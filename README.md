
# Polarization-identification-on-temporal-graphs-in-Eulerian-description

## Initial setting

### Downloads

Download the 'code' file of the present repository. Codes from TGN (https://github.com/twitter-research/tgn) are reused here for the sake of compatibility.
Compare to the original repository, the following files were replaced or added:
 - For TGN:
   - 'train_self_supervised.py';
   - 'utils/preprocess_data.py';
   - 'utils/data_processing.py';
   - 'model/tgn.py';
   - 'evaluation/evaluation.py';
   - 'modules/embedding_module.py';
   - 'TGN_vector_fields.ipynb'.

### Datasets

Three datasets are provided:
  - politoEdge_graph_t_0.12_d_1;
  - politoEdge_graph_t_0.3_d_1;
  - politoEdge_graph_t_1_d_1.

They correspond to the same original dataset, but on more or less important time ranges.

## Running the codes

Open the code/TGN_vector_fields.ipynb notebook. Run all to test it on the politoEdge_graph_t_0.3_d_1 dataset. Results are available in Part. B. and Part. C. of the notebook.

The model provided has been trained for t=0.12. To use it over a different time range, change the title of the data to be studied in Part. C.
