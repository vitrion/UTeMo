# 🎭 UTeMo: Audiovisual Emotion Dataset for Mexican Spanish

UTeMo (Universidad Tecnológica de la Mixteca Emotion Dataset) is a high-quality audiovisual dataset designed for **multimodal emotion recognition** using **facial expressions and speech signals** in Mexican Spanish. It is specifically tailored for machine learning and deep learning research in affective computing.

## 📌 Overview

UTeMo contains **1801 validated video samples** representing the six basic emotions defined by Ekman (anger, sadness, happiness, surprise, fear, and disgust) plus a neutral state. Each sample includes synchronized **high-resolution video** and **high-fidelity audio**, making it suitable for multimodal approaches.

- 🎥 Video resolution: 1920×1080 (30 fps)  
- 🔊 Audio sampling rate: 48 kHz  
- 👥 Actors: 14 (7 female, 7 male)  
- 🗣 Language: Mexican Spanish  
- ⏱ Total duration: ~105 minutes  

## 📥 Dataset Download

The dataset is publicly available in Mendeley Data:

👉 [Download UTeMo dataset from Mendeley](https://data.mendeley.com/datasets/x5rmd28h73/1)

After downloading, extract all `.avi` files into:

```bash
UTeMo/data/UTeMo_video/
```

## 🏷️ File Naming Convention

Each video file follows the naming pattern:

**Emotion**_Actor\_**N**\_P\_**M**\_F\_**L**.avi

Where:
- **Emotion** → Emotional state (Anger, Sadness, Happiness, Surprise, Fear,
  Disgust, Neutral)
- **N** → Actor ID (1--14)
- **M** → Repetition index (1--3)
- **L** → Phrase ID

**✅ Example**

`Happiness_Actor_1_P_2_F_5.avi`

This indicates:
- Emotion: Happiness
- Actor: 1
- Repetition: 2
- Phrase: 5

## 📁 Repository Structure

- `UTeMo`:
	- `data`:
		- `UTeMo_video` \<\-\-- Place downloaded AVI files here
	- `preprocessing`:
		- `1_audio_extraction.py`
		- `2_audio_denoising.py`
		- `3_audio_nonsilent_detection.py`
		- `4_audio_mel_spectrogram_generation.py`
		- `5_stratified_emotion_splitter.py`
		- `6_merge_train_val.py`
		- `7_leave_actor_out_partitioning.py`
		- `8_data_augmentation`
		- `noise.wav`
	- `supplementary_files`:
		- `figures`:
			- `fig4_3d_emotion_counts.png`
			- `fig5_polygon_categorical_proportions.png`
			- `fig6_pie_overall_intensity_likert5.png`
			- `fig7_grouped_intensity_by_emotion_likert5.png`
		- `compute_kappa_of_fleiss.py`
		- `plot_emotions.py`
		- `real_emotions.csv`
		- `sample_class_distribuition.csv`
		- `spanish_phrases.pdf`
		- `SurveyResponses.csv`
	- `training_eval`:
		- `figures`:
			- `fser_training_curves_60_20_20.png`
		- `logs`:
			- `fser_coarse_60_20_20.csv`
     		- `fser_final_test_log_60_20_20.txt`
			- `fser_fine_60_20_20.csv`
			- `fser_leave_actor_out_results.csv`
			- `fser_training_history_60_20_20.csv`
		- `models`:
			- `__pycache__`
			- `model_eval.py`
		- `9_trainer.py`
		- `10_leave_actor_out.py`
	- `LICENSE`
	- `README.md`
	- `requirements.txt`
	
## 💥 UTeMo FSER weights

With the aim of making the Keras weight file of the FSER model presented in this research work publicly available, it can be accessed through the following link:

👉 [Download UTeMo FSER weights](https://1drv.ms/u/c/26b45e0f43f4667b/IQAu3da7ZBnFTLgfNjCsub5jAZCliyxAZ9zsOT7LuOWyggs?e=9BZfa5)

## ⚙️ Usage Instructions

To reproduce preprocessing and training:

**Step 1: Run preprocessing scripts**

Execute all scripts in the `./preprocessing` folder **in numerical order**:

`1_*.py` → `2_*.py` → `3_*.py` → `4_*.py` → `5_*.py` → `6_*.py` → `7_*.py` → `8_*.py`

These scripts will generate all required processed data inside the `./data`
directory.

The scripts `5_stratified_emotion_splitter.py`, `6_merge_train_val.py`, and `8_data_augmentation.py` require selecting a specific dataset partitioning scheme by modifying the global variable `AUDIO_PART_SEL`. To reproduce the results reported in the article, this variable must be set as follows:

- `PART_SEL = '60_20_20'` for FSER experiments.

**Step 2: Train models**

After preprocessing, run:
- `./training_eval/9_trainer.py`

This script train a deep learning model using audio (FSER) representations. Also, it requires selecting the partitioning scheme by setting the global variable `PART_SEL` using the same configuration described above. Additionally, this script must be executed in four sequential stages controlled by the variable `stage`:

1. `stage = 'coarse'`, which performs a coarse grid search and stores results in a CSV file in the `./logs` directory; 
2. `stage = 'fine'`, which performs a refined search over the number of epochs using the best configuration from the previous stage;
3. `stage = 'train'`, which trains the model using the best hyperparameters from the fine search and generates training curves (accuracy and loss); and
4. `stage = 'test'`, which conducts final training using a merged `train+val` set, saves the trained model in the `./models` directory (in Keras format), and outputs evaluation metrics (accuracy and F1-score) into a log file.

These stages must be executed sequentially to fully reproduce the experimental pipeline.

**Step 3: Leave one actor out validation (optional)**

Optionally, after running the trainer script, you should run:
- `./training_eval/10_leave_actor_out.py`

This script will perform a leave-one-actor cross-validation, where all samples from each actor are separated into a test set, and the deep learning model is trained using all samples from the other actors. This procedure aims to evaluate the generalizability of the predictive model, ensuring that the system learns universal characteristics of emotions and not just the tone of voice of a particular actor. The results of this exercise are saved to a CSV file in the `./logs` directory.

## 👨‍🔬 Authors

- Jorge-Arturo Carrasco-Jiménez (first autor): <carrascojj.106@gmail.com>
- Arturo Téllez-Velázquez: <atellezv@mixteco.utm.mx>
- Rosebet Miranda-Luna: <rmiranda@mixteco.utm.mx>
- Antonio Orantes-Molina: <tonito@mixteco.utm.mx>
- Juan-Pablo García-Vázquez: <pablo.garcia@uabc.edu.mx>
- Raúl Cruz-Barbosa (corresponding author): <rcruz@mixteco.utm.mx>

## 📖 Citation

If you use UTeMo in your research, please cite:

Carrasco-Jiménez, J. A., Téllez-Velázquez, A., Miranda-Luna, R.,
Orantes-Molina, A., García-Vázquez, J. P., & Cruz-Barbosa, R. (2026). An
audiovisual dataset for emotion recognition (UTeMo) in Spanish language.
Data in Brief, in revision.

## 💡 Applications

UTeMo can be used in:
- Emotion recognition systems
- Human-computer interaction
- Affective computing
- Speech and facial analysis
- Psychology and behavioral studies


## ⚠️ Limitations

- Limited number of actors (14)
- Only basic Ekman emotions + neutral
- Limited demographic diversity

## 🧾 License & Ethics

All participants signed informed consent forms, and no personally
identifiable information is included in the dataset. The dataset is
intended strictly for research and academic purposes.


## 🙌 Acknowledgments

We thank the Universidad Tecnológica de la Mixteca (UTM), the Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI), psychologist Ivette Jimenez García, and all volunteer actors and raters in México who contributed to this dataset.
