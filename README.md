
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

The dataset provided, corresponding to the first ten percent of the one used for the paper, is noted politoEdge_graph_t_0.1_d_1.
The interaction instants have been noised.

## Running the codes

Open the code/TGN_vector_fields.ipynb notebook. Run all to test it. Results are available in Part. B. and Part. C. of the notebook.
