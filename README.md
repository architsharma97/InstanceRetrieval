# Instance Retrieval using Vocabulary Tree on VGG16 feature space
Submission towards partial completion of course requirements of CS698O: Visual Recognition. The task was to create a model to retrieve instances of training data classes from unseen data.

**Model**: Propose Regions using [Selective Search](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib) (No threshold on the minimum size). The selective search implementation was picked up from [AlpacaDB](https://github.com/AlpacaDB/selectivesearch).
Pass the proposed regions through VGG16 and construct a feature space. Pre-trained model was picked up from [Keras](https://keras.io/).
Construct a _Hierarchical K-Means_ based _Vocabulary Tree_, along the lines of this [paper](http://www-inst.eecs.berkeley.edu/~cs294-6/fa06/papers/nister_stewenius_cvpr2006.pdf).

To test a new image, create text file containing the addresses of the image relative to the 'Scripts/' folder.
Assuming the models have been saved in the 'Models/' folder, run the following command from the Scripts folder for the image 'Test/test.jpg', which is accompanied by 'Test/address.txt'

```
python query.py Test/address.txt ../Models/PCAlayer_128.pkl ../Models/vocab_tree_20_5/vocab_tree_20_5.pkl ../train_list.txt
```
This produces 'Test/test.txt' which contains matched images in the training set.
The PCA Matrix and Vocabulary Tree models can be chosen from the Models folder.