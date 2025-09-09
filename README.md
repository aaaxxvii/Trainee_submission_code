# 株価予測モデルの構築

## 概要 (Description)
本プロジェクトでは、株価データ (`stock_price.csv`) を用い、株価予測モデルを構築・評価します。
 trainee_EDA.ipynbにて、データの傾向や相関を視覚的に分析、trainee.pyで予測モデルの構築・学習・評価を行います。
 
## 目次 (Table of Contents)

- [環境](#環境)
- [インストール](#インストール)
- [使用法](#使用法)

## 環境
このプロジェクトで使用した主要なライブラリやツールです。

trainee_EDA.ipynbに必要
- **Python 3.12**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Japanize-matplotlib**

trainee.pyに必要
- **Python 3.12**
- **TensorFlow / Keras**
- **Optuna**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**



## インストール

このプロジェクトをローカルで実行するための手順です。

1. **リポジトリをクローン**
    ```sh
    git clone https://github.com/aaaxxvii/Trainee_submission_code.git
    ```
2. **必要なライブラリをインストール**

   以下のコマンドで、必要なライブラリをインストールします。
    ```sh
    pip install pandas numpy matplotlib scikit-learn tensorflow optuna japanize-matplotlib
    ```


## 使用法

### trainee_EDA.ipynbについて
- ローカルで実行する場合、`stock_price.csv`をコードファイルと同じディレクトリに配置し、VScodeやJupyterなどで実行
- Google Colab でも簡単に実行できます。

### trainee.pyについて
1. **データセットの準備**: `stock_price.csv`をコードファイルと同じディレクトリに配置してください。
2. **学習の実行**: 以下のコマンドで、モデルの学習、ハイパーパラメータ最適化、および評価が実行されます。
    ```sh
    python trainee.py
    ```
