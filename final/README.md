# ML2019FALL Final Project - Domain Adaptation

Team name: NTU_r08521610_rainforest
Team member: r08521610鄭羽霖 r07521603蔡松霖 r08521602王鈞平

## Conda environment

```shell
conda env create -f environment.yml
```

## Download models (共8組，每組包含G，C1，C2)

下載model，執行download.sh:

```shell
./download.sh
```

## Predict (Reproduce)

後面data的部分需要依照data所在路徑做修正，以`./data`為例：

```shell
python3 ./src/new_pred.py ./data/trainX.npy ./data/trainY.npy ./data/testX.npy ./checkpoints/

python3 ./src/output.py
```

過程會產出一些csv檔於`./pred`資料夾中，最後ensemble的結果為`./pred/ensemble.csv`

## Train

使用`./src`下的`MCD.py`，同上須針對data的路徑做修改，後面的參數依序是batch_size, epoch, save_epoch, num_k, optimizer('adam' or 'momentum'), lr

model會存在`./checkpoint`
training loss會記錄在`./record/k_4_0.csv` (隨跑job次數更新命名第二數字)

```shell
python3 ./src/MCD.py ./data/trainX.npy ./data/trainY.npy ./data/testX.npy 256 800 10 4 adam 2.5e-4
```

如需產生csv檔預測的label，可依下面範例執行`./src/pred.py`，其中`./checkpoint`可視存放model的地方做調整，會將所有存放在該路徑下的model做predict的動作，存放至`./pred`：

```shell
python3 ./src/pred.py ./data/trainX.npy ./data/trainY.npy ./data/testX.npy ./checkpoint
```
