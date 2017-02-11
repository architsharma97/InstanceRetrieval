Submission towards partial completion of course requirements of CS698O: Visual Recognition. The task was to create a model to retrieve instances of training data classes from unseen data.

Model: Propose Regions using [Selective Search](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)(No threshold on the minimum size). The selective search implementation was picked up from [AlpacaDB](https://github.com/AlpacaDB/selectivesearch).
Pass the proposed regions through VGG16 and construct a feature space. Pre-trained model was picked up from [Keras](https://keras.io/).
Construct a hierarchical k-means based vocabulary tree, along the lines of this [paper](http://www-inst.eecs.berkeley.edu/~cs294-6/fa06/papers/nister_stewenius_cvpr2006.pdf).

To test a new image, create text file containing the addresses of the image relative to the 'Scripts/' folder.
Assuming the models have been saved in the 'Models/' folder, run the following command from the Scripts folder 

```
python query.py /path/to/address/file.txt ../Models/PCAlayer_128.pkl ../Models/vocab_tree.pkl ../train_list.txt
```
The PCA Matrix and Vocabulary Tree models can be chosen from the Models folder.