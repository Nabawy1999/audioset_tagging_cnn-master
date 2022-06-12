from(Analyzing the Potential of Pre-Trained Embeddings for Audio Classification Tasks)paper
///—In the context of deep learning, the availability of
large amounts of training data can play a critical role in a model’s
performance. Recently, several models for audio classification
have been pre-trained in a supervised or self-supervised fashion
on large datasets to learn complex feature representations, socalled embeddings. These embeddings can then be extracted
from smaller datasets and used to train subsequent classifiers. In
the field of audio event detection (AED) for example, classifiers
using these features have achieved high accuracy without the
need of additional domain knowledge.

///-With the availability of large audio datasets in recent years
(e.g. AudioSet [1]), many audio classification tasks based on
deep learning techniques have seen improved classification
accuracy. This has mostly occurred in scenarios where data is
abundant or easily collected, such as speech or environmental
sounds. However, for many audio classification tasks, large
scale data collection is unrealistic. One such example in the
field of Music Information Retrieval (MIR) is classification
of non-western music (e.g. regional traditional music). For
musicological analysis, automatic classification can be a powerful tool; however, performing annotations at large scale
is restricted, among other things, by the amount of domain
knowledge required for the annotations. A similar problem
arises in the field of Industrial Sound Analysis (ISA) for
acoustic quality control applications [2]. The goal is to assess
the health of a given machine by analyzing the sound it
produces. However, large amounts of training examples are very costly to obtain for every product, machine, and possible
fault. With this in mind, this work focuses on training and
evaluating a number of ISA and MIR classifiers, under the
premise that only a small amount of annotated training data
will be available.

///-Transfer Learning (TL) is a powerful technique for building
classifiers for small datasets. The main idea behind TL is to
pre-train models on tasks where data is abundant, and re-use
the knowledge gained during training for tasks where data is
limited [3]. There are two main TL approaches *cite the paper* : In the first
approach, a trained model (obtained with a large dataset) is
fine-tuned on the task-specific dataset. In the second approach,
learned feature representations, also called embeddings, are
used to train additional classifiers on task-specific datasets. TL
was shown to be a promising training strategy for a variety
of research fields such as Image Classification [4], Natural
Language Processing [5], Environmental Sound Classification
(ESC) [6]–[8], and several MIR tasks like genre classification
[9] and instrument recognition [10].

-Results obtained with the embeddings
are compared to baseline systems where the classifiers are
trained from scratch using only the task-specific dataset.
-In [9], a VGG network architecture (originally proposed
for image classification in [13]) was modified and trained in a supervised fashion with audio data from the Million
Song Dataset [14]. On all the evaluated tasks, the embeddings outperformed a baseline using Mel Frequency Cepstral Coefficients (MFCCs) as input representation. However,
performance was still below the state-of-the-art (except for
speech/music classification where nearly perfect classification
was achieved with all methods).

-((MODIFICATION)): 2) Task 2: Musical Instrument Family Recognition (T2):
For the task of instrument family recognition, we use an
in-house dataset called DB-MTC [20], which contains 50
commercial recordings of different composers of Western
classical music. Each recording is a polyphonic piece of
music composed for one instrument family and is, hence,monotimbral. The five instrument families are woodwinds,
brass, piano, vocal, and strings. As a baseline, we intended to
use the instrument recognition CNN model proposed in [21].
However, this model showed a tendency to overfit on DBMTC leading to a classification accuracy of 72%. (((( Therefore,
we removed two of the original four convolutional blocks to
decrease the number of trainable parameters to 10%. With this
modification, state-of-the-art performance was achieved with
a file-wise accuracy of  . )))) Full details about the baseline
model can be found in the study website.3

-((MODIFICATION)): Task 3: Speech Music Classification (T3): This task is an
extended version of speech/music classification which considers four classes: speech, solo singing, choir, and instrumental
music. The dataset of ethnomusicological field recordings
from [22] is used here.4
Initially, the model proposed in
[22] was considered as a baseline. This model achieved a
final accuracy of 94% with ((((*data enrichment* from several
speech/music datasets and *augmentation via pitch shifting and
time stretching*. Additionally, [22] reports that a multilayer
perceptron with 16 units trained on VGGish embeddings
achieved 86.7%. For comparability between the other tasks
of this paper, no additional augmentation methods or other
datasets were used. The reduced amount of training data led
to overfitting and unstable training with the initial model from
[22].3 Therefore, the smaller CNN architecture from [19] is
used as baseline, leading to a file-wise accuracy of 88.6%.

-((MODIFICATION)): Task1: The best
performing model reported in [19] is used as baseline, where
the best classification result (80.7% file-wise accuracy) was
obtained with a *feedforward* neural network.
-VGGish, Kumar, and OpenL3 embeddings were chosen
for this study since they have already shown state-of-the-art
performance on the ESC50 dataset. VGGish are embeddings
trained in a supervised fashion with weak labels, the Kumar
embeddings are trained in a supervised fashion considering
weak labels and with improved annotations from AudioSet,
and OpenL3 are self-supervised embeddings learned from
AudioSet.

----------------------------------------------------------------------------

From(Transfer learning and the art of using Pre-trained Models in Deep Learning)article

-// “Transfer Learning” which enables us to use pre-trained models from other people 
  by making small changes. In this article, I am going to tell how we can use pre-trained models
  to accelerate our solutions.

-A neural network is trained on a data. This network gains knowledge from this data, 
which is compiled as “weights” of the network. 
These weights can be extracted and then transferred to any other neural network. 
Instead of training the other neural network from scratch, we “transfer” the learned features.

(SEE THE DIAGRAM)
-//transfer learning by passing on weights is equivalent of language used to disseminate knowledge over 
generations in human evolution.
-//Simply put, a pre-trained model is a model created by some one else to solve a similar problem. 
Instead of building a model from scratch to solve a similar problem, 
you use the model trained on other problem as a starting point. A pre-trained model may not be 100% accurate in your application, 
but it saves huge efforts required to re-invent the wheel.

-//What is our objective when we train a neural network? We wish to identify the correct 
weights for the network by multiple forward and backward iterations. By using pre-trained models which 
have been previously trained on large datasets, we can directly use the weights and 
architecture obtained and apply the learning on our problem statement. This is known as transfer learning. 
We “transfer the learning” of the pre-trained model to our specific problem statement.

-//You should be very careful while choosing what pre-trained model you should use in your case. If the problem statement 
we have at hand is very different from the one on which the pre-trained model was trained – the prediction 
we would get would be very inaccurate. For example, a model previously trained 
for speech recognition would work horribly if we try to use it to identify objects using it.

*PARAPHRASE BY HAND*
-//We are lucky that many pre-trained architectures are directly available for us in the Keras library. 
Imagenet data set has been widely used to build various architectures since it is large enough (1.2M images) 
to create a generalized model. The problem statement is to train a model that can correctly classify 
the images into 1,000 separate object categories. These 1,000 image categories represent object classes that we 
come across in our day-to-day lives, such as species of dogs, cats, various household objects, vehicle types etc.

-//These pre-trained networks demonstrate a strong ability to generalize to images outside the ImageNet dataset via transfer learning. 
We make modifications in the pre-existing model by fine-tuning the model. Since we assume that the 
pre-trained network has been trained quite well, we would not want to modify the weights too soon and too much. 
While modifying we generally use a learning rate smaller than the one used for initially training the model.

-Ways to Fine tune the model: (SEE THE DIAGRAM)
1-Feature extraction – We can use a pre-trained model as a feature extraction mechanism. What we can do is that 
we can remove the output layer( the one which gives the probabilities for being in each of the 1000 classes) 
and then use the entire network as a fixed feature extractor for the new data set.
2-Use the Architecture of the pre-trained model – What we can do is that we use architecture of the model 
while we initialize all the weights randomly and train the model according to our dataset again.
3-Train some layers while freeze others – Another way to use a pre-trained model is to train is partially. 
What we can do is we keep the weights of initial layers of the model frozen while we retrain 
only the higher layers. We can try and test as to how many layers to be frozen and how many to be trained.

-1 hena ma3 2 hnak
 3 ma3 3
 


-Scenario 1 – Size of the Data set is small while the Data similarity is very high – In this case, since the data 
similarity is very high, we do not need to retrain the model. All we need to do is to customize and modify the out
put layers according to our problem statement. We use the pretrained model as a feature extractor. Suppose we 
decide to use models trained on Imagenet to identify if the new set of images have cats or dogs. Here the images 
we need to identify would be similar to imagenet, however we just need two categories as my output – cats or dogs. 
In this case all we do is just modify the dense layers and the final softmax layer to output 2 categories instead
of a 1000.
	Scenario 2 – Size of the data is small as well as data similarity is very low – In this case we can freeze the 
initial (let’s say k) layers of the pretrained model and train just the remaining(n-k) layers again. The top 
layers would then be customized to the new data set. Since the new data set has low similarity it is significant 
to retrain and customize the higher layers according to the new dataset.  The small size of the data set is 
compensated by the fact that the initial layers are kept pretrained(which have been trained on a large dataset 
previously) and the weights for those layers are frozen.
	Scenario 3 – Size of the data set is large however the Data similarity is very low – In this case, since we have 
a large dataset, our neural network training would be effective. However, since the data we have is very different
as compared to the data used for training our pretrained models. The predictions made using pretrained models 
would not be effective. Hence, its best to train the neural network from scratch according to your data.
	Scenario 4 – Size of the data is large as well as there is high data similarity – This is the ideal situation. 
In this case the pretrained model should be most effective. The best way to use the model is to retain the 
architecture of the model and the initial weights of the model. Then we can retrain this model using the weights 
as initialized in the pre-trained model.

----------------------------------------------------------------------------------------

From(WAVEFORMS AND SPECTROGRAMS: ENHANCING ACOUSTIC SCENE CLASSIFICATION
USING MULTIMODAL FEATURE FUSION)paper

-(((Intro))): Acoustic scene classification (ASC) has seen tremendous progress from the combined use of convolutional 
neural networks (CNNs) and signal processing strategies.

-Mel-spectrograms are the de-facto audio feature representation and
have been widely used throughout the history of audio understanding [1]. Mel-spectrograms are created by calculating the shorttime fourier transform (STFT) of an audio signal, then passing the
STFT frequency responses through band-pass filters spaced on the
Mel(logarithmic)-scale and often further passed through a logarithmic compression to replicate the human’s non-linear perception of
signal pitch and loudness, respectively.
With the advent of deep neural networks, many methods have
been introduced that perform audio understanding tasks such as
ASC, audio tagging, and sound event detection by using Melspectrogram representations of audio as the input to a convolutional
neural network [2, 3].

-Researchers have also explored other feature representations such as the gammatone and Constant-Q (CQT)
spectrogram, and Mel Frequency Cepstrum Coefficients (MFCC)
[4, 5]. [6] and found that fusing these representations allows for a
network to learn complementary features, creating a stronger model
for ASC.

- Researchers are now looking at hybrid methods
that use both waveform and spectrogram representations in a fusion
setting(cite this paper). In this paper we perform early feature map fusion of waveform and
spectrogram features that are passed through convolutional layers
for audio tagging and environmental sound classification

---------------------------------
((THE DCASE 2021 CHALLENGE TASK 6 SYSTEM: AUTOMATED AUDIO CAPTIONING
WITH WEAKLY SUPERVISED PRE-TRAING AND WORD SELECTION METHODS))paper

 -Wavegram-Logmel-CNN use CNN14
as a backbone architecture on the extracted Wavegram and logmel
features, where Wavegram are extracted from time-domain waveforms by trainable one-dimensional CNN followed by three convolutional
blocks. (also cite) (WAVEFORMS AND SPECTROGRAMS: ENHANCING ACOUSTIC SCENE CLASSIFICATION
USING MULTIMODAL FEATURE FUSION)

------------------------------------------------------------------------------------
((Data Augmentation))
//-Acoustic environments affect acoustic characteristics of sound to be
recognized by physically interacting with sound wave propagation.
Thus, training acoustic models for audio and speech tasks requires
regularization on various acoustic environments in order to achieve
robust performance in real life applications.

//- Some of data augmentation methods proposed for computer vision tasks such as mixup [9] are actively adopted in audio
and speech domain. However, most of the image data augmentation methods including rotation, flip, shear and crop [1] result in
irrelevant transform of audio data when applied on spectrograms.Therefore, data augmentation methods consistent with acoustics and
signal processing domain knowledge are required to effectively train
acoustic models in audio and speech domain. 

//-Data augmentation methods in audio and speech domain includes
conventional audio signal processing methods such as time stretching, pitch shift, clipping, suppressing, adding noise, adding reverberation, etc.These methods reflect domain knowledge in acoustics and signal processing, thus they have been frequently adopted
for data augmentation purpose.However, data augmentation using
conventional audio signal processing methods could introduce some
inefficiencies when training acoustic models. Applying conventional
audio signal processing methods requires prior knowledge to appropriately handle audio data. In addition, these methods may involve
more computations in expense for more natural sound, which does
not even guarantee to train acoustic models better. Such inefficiencies hinder optimal training of acoustic models. Therefore, we need
data augmentation methods that are simple, intuitive, yet effective
for training acoustic models to learn to extract information from audio data.

-SpecAugment [10] is one of the most powerful and widely used
data augmentation methods in audio and speech domain. Instead of
applying data augmentation on waveform, it proposed time warping,
time masking, and frequency masking those could be directly applied on log mel spectrogram. As it is applied directly on the input
feature space, it is easy to comprehend and use. Intuitively, applying
time warping on audio would sound like the audio played faster in
some points and slower in some other points. Time masking would
sound that some parts are not played for short duration. Frequency
masking would sound like some part of frequency range is missing.
As long as these distortions are not too severe, human can recognize
the content of audio data after these processing, and trained acoustic
models should do as well. Although these methods do not sound as
natural as conventional audio processing methods when transformed
back in waveform, it helps training acoustic models more effectively
with extreme cases

-As frequency masking removes information from
certain random frequency range, it helps to train acoustic model to
infer the sound information from less distinctive frequency regions
too. However, frequency masking completely removes certain part
of energy that might help inferring the sound information. Such
brutal damage on spectrum not only rarely happens in real situations but also causes the model to be trained to forcibly extract
information from indistinct frequency ranges. Therefore, FilterAugment weakens some parts of frequency range while strengthening
other parts instead. Lowering energy instead of removing it would
at least let acoustic models to infer the information from that frequency region. In addition, increasing other frequency range energy
would train acoustic models to recognize sound information from
various frequency region as they will be trained with the same data
highlighted on different frequency region every epoch. Therefore,
FilterAugment helps training acoustic models to extract information
from the wider range of frequency regardless of each frequency’s
relative significance composing the sound information.

-

----------------------------------------------------------------------------------------------------------------------
(((((((((((((((((METHODOLOGY))))))))))))))))):
-Unlike previous work, Audio Set considers all sound events
rather than a limited domain. We believe that a large-scale task, in
terms of both categories and data, will enable more powerful learning techniques, and hence a step-change in system quality.


















general talk:
-The choice of activation functions in deep networks has a significant effect on
the training dynamics and task performance. Currently, the most successful and
widely-used activation function is the Rectified Linear Unit (ReLU). Although
various hand-designed alternatives to ReLU have been proposed, none have managed to replace it due to inconsistent gains.

-The activation function plays a major role in the success of training deep neural networks. Currently, the most successful and widely-used activation function is the Rectified Linear Unit (ReLU) (Hahnloser et al., 2000; Jarrett et al., 2009; Nair & Hinton, 2010), defined as
f(x) = max(x, 0). The use of ReLUs was a breakthrough that enabled the fully supervised training
of state-of-the-art deep networks (Krizhevsky et al., 2012)


@article{ramachandran2017searching,
  title={Searching for activation functions},
  author={Ramachandran, Prajit and Zoph, Barret and Le, Quoc V},
  journal={arXiv preprint arXiv:1710.05941},
  year={2017}
}






why leaky relu:

LReLU enables a small amount of information to flow when x < 0

@article{ramachandran2017searching,
  title={Searching for activation functions},
  author={Ramachandran, Prajit and Zoph, Barret and Le, Quoc V},
  journal={arXiv preprint arXiv:1710.05941},
  year={2017}
}



-The leaky rectifier allows for a small, non-zero gradient when the unit
is saturated and not active,Figure 1 shows the LReL function, which is nearly
identical to the standard ReL function. The LReL
sacrifices hard-zero sparsity for a gradient which is potentially more robust during optimization

@inproceedings{maas2013rectifier,
  title={Rectifier nonlinearities improve neural network acoustic models},
  author={Maas, Andrew L and Hannun, Awni Y and Ng, Andrew Y and others},
  booktitle={Proc. icml},
  volume={30},
  number={1},
  pages={3},
  year={2013},
  organization={Citeseer}
}

-Leaky ReLU [18], like ReLU, is also equivalent to the identity function for positive values but has a hyperparameter 𝛼 > 0 applied to the negative
inputs to ensure the gradient is never zero. As a result, Leaky ReLU is not as prone to
getting caught in local minima and counters ReLU's problem with hard zeros that makes
it more likely to fail to activate.

@article{nanni2021comparison,
  title={Comparison of different convolutional neural network activation functions and methods for building ensembles},
  author={Nanni, Loris and Maguolo, Gianluca and Brahnam, Sheryl and Paci, Michelangelo},
  journal={arXiv preprint arXiv:2103.15898},
  year={2021}
}

why mish:

-Mish is almost smooth at any point on
the curve, allowing the transformation of more valid information into the model to improve
accuracy and generalization performance.
-A new activation function referred to as Mish has recently been widely used in deeplearning methods and is described as
(((((figure))))))
- In Mish, the derivative of a continuous order is infinite, which
is an advantage over ReLU. The continuous order of ReLU is 0, indicating that it is not
continuously differentiable, which can potentially present problems in gradient-based
optimization.

@article{wang2022smish,
  title={Smish: A Novel Activation Function for Deep Learning Methods},
  author={Wang, Xueliang and Ren, Honge and Wang, Achuan},
  journal={Electronics},
  volume={11},
  number={4},
  pages={540},
  year={2022},
  publisher={MDPI}
}


why swish


-experiments show that the best discovered activation function,
f(x) = x · sigmoid(βx), which we name Swish, tends to work better than ReLU
on deeper models across a number of challenging datasets, Swish consistently matches or outperforms ReLU on deep networks applied to a variety
of challenging domains such as image classification and machine translation. For example, simply
replacing ReLUs with Swish units improves top-1 classification accuracy on ImageNet by 0.9% for Mobile NASNet-A (Zoph et al., 2017) and 0.6% for Inception-ResNet-v2 (Szegedy et al., 2017),These accuracy gains
are significant given that one year of architectural tuning and enlarging yielded 1.3% accuracy improvement going from Inception V3 (Szegedy et al., 2016) to Inception-ResNet-v2 (Szegedy et al.,
2017). The simplicity of Swish and its similarity to ReLU make it easy for practitioners to
replace ReLUs with Swish units in any neural network.

@article{ramachandran2017searching,
  title={Searching for activation functions},
  author={Ramachandran, Prajit and Zoph, Barret and Le, Quoc V},
  journal={arXiv preprint arXiv:1710.05941},
  year={2017}
}


The simplicity of Swish and its similarity to ReLU make it easy for practitioners to
replace ReLUs with Swish units in any neural network.










-Audio pattern recognition is an important research topic
in the machine learning area, and plays an important role
in our life. We are surrounded by sounds that contain rich
information of where we are, and what events are happening
around us. Audio pattern recognition contains several tasks
such as audio tagging [1], acoustic scene classification [2],
music classification [3], speech emotion classification and
sound event detection [4].

-To address this issue, in 2017, Google released AudioSet
[2]. This dataset contains 2.1 millions of 10s audio sound grabbed
from YouTube videos and annotated with presence / absence
labeling of 527 types of sound events.

@article{arnault2020crnns,
  title={CRNNs for Urban Sound Tagging with spatiotemporal context},
  author={Arnault, Augustin and Riche, Nicolas},
  journal={arXiv preprint arXiv:2008.10413},
  year={2020}
}



--Wavegram-Logmel-CNN use CNN14
as a backbone architecture on the extracted Wavegram and logmel
features

--Unlike previous work, Audio Set considers all sound events
rather than a limited domain. 

-The choice of activation functions in deep networks has a significant effect on
the training dynamics and task performance.

-Audio tagging is an active research area and has
a wide range of applications. Since the release of AudioSet,
great progress has been made in advancing model performance,
which mostly comes from the development of novel model
architectures and attention modules. However, we find that appropriate training techniques are equally important for building
audio tagging models with AudioSet, but have not received
the attention they deserve

--Audio tagging aims to identify sound events that occur in
a given audio recording, and enables a variety of Artificial
Intelligence-based systems to disambiguate sounds and understand the acoustic environment. Audio tagging has a wide
range of health and safety applications in the home, office,
industry, transportation, and has become an active research
topic in the field of acoustic signal processing

@article{gong2021psla,
  title={Psla: Improving audio tagging with pretraining, sampling, labeling, and aggregation},
  author={Gong, Yuan and Chung, Yu-An and Glass, James},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={3292--3306},
  year={2021},
  publisher={IEEE}
}








Audio tagging has a wide
range of health and safety applications in the home, office,
industry, transportation, and has become an active research
topic in the field of acoustic signal processing