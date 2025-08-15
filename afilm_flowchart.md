# AFILM Audio Super-Resolution - Flowchart

## 1. Data Loading (`codes/utils.py`)

**Prétraitement** :
- **Chargement** : `load_h5(h5_path)` lit les fichiers `.h5` du dataset VCTK
  - `'data'` (patches LR) et `'label'` (patches HR)
- **Upsampling initial** : `spline_up(x_lr, r)` - interpolation spline 
- **Découpage en patches** : 
  - Padding pour multiple de `patch_size` (défaut: 8192)
  - Reshape en `(n_patches, patch_size, 1)`
  - Downsampling par facteur `r` (2, 4, ou 8)

**Entrée pour le modèle** : 
- Tenseurs audio LR de shape `[batch, 8192, 1]`
- Tenseurs audio HR correspondants pour l'entraînement

---

## 2. Model Definition (`codes/models/afilm.py`)


### Couches utilisées :
- **AFiLM Layer** (`codes/models/afilm.py`) :
  - Adaptive Feature-wise Linear Modulation
  - Utilise un **TransformerBlock** (4 couches, 8 têtes d'attention)
  - Normalisation par blocs avec pooling temporel
- **TransformerBlock** (`codes/models/layers/transformer.py`) :
  - Multi-Head Self-Attention (8 têtes)
  - Position encoding sinusoïdal
  - Feed-forward network (2048 dimensions cachées)
- **SubPixel1D** (`codes/models/layers/subpixel.py`) :
  - Upsampling par réorganisation des dimensions
  - Facteur 2 à chaque étape

### Structure du modèle :
1. **Downsampling** (4 couches) :
   - Conv1D → MaxPooling1D → LeakyReLU → AFiLM
   - Filtres : [128, 256, 512, 512]
   - Kernel sizes : [65, 33, 17, 9]
2. **Bottleneck** :
   - Conv1D → MaxPooling1D → Dropout(0.5) → AFiLM
3. **Upsampling** (4 couches) :
   - Conv1D → ReLU → SubPixel1D → AFiLM → Skip connections
4. **Output** :
   - Conv1D finale + SubPixel1D + connexion résiduelle

**Entrée/Sortie** :
- Input : patch audio LR `[batch, 8192, 1]`
- Output : patch audio HR reconstruit `[batch, 8192, 1]`

---

## 3. Training Loop (`codes/train.py`)

**Configuration** :
- **Loss** : `'mse'` 
- **Optimizer** : `Adam` (learning rate par défaut : `3e-4`)
- **Métrics** : `RootMeanSquaredError()`
- **Batch size** : 16 (par défaut)
- **Epochs** : 20 (par défaut)

**Boucle d'entraînement** :
```python
# Chargement des données
X_train, Y_train = load_h5(args.train)
X_val, Y_val = load_h5(args.val)

# Compilation du modèle
model.compile(optimizer=Adam(lr=3e-4), loss='mse', 
              metrics=[RootMeanSquaredError()])

# Entraînement
model.fit(X_train, Y_train, 
          batch_size=16, epochs=20,
          callbacks=[CustomCheckpoint])
```


- `CustomCheckpoint` : sauvegarde automatique du modèle à chaque époque

---

## 4. Evaluation / Inference (`codes/test.py`)

1. **Charge modèle** :
   - `keras.models.load_model()` avec objets custom AFiLM/TFiLM
2. **Fonction principale** : `upsample_wav(wav_file, args, model)`
   - Charge un fichier audio complet avec `librosa.load()`
   - Applique prétraitement (padding, downsampling, découpage en patches)
   - Prédit : `model.predict(patches, batch_size=16)`
   - Reconstruit le signal complet

**Sorties** :
- **Audio reconstruit HR** : `.pr.wav` (prédiction)
- **Audio original HR** : `.hr.wav` (ground truth)  
- **Audio LR** : `.lr.wav` (entrée basse résolution)

---