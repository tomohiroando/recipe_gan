# recipe_gan
StackGANによるtext-to-imageを料理のレシピから行ったらどうなるのかを実験した
## 動作環境
Python 3.6
## どんな流れ
* レシピのテキストから体言と用言を抽出
* Doc2Vecで文書ベクトル化
* モデルで学習

## データセット
楽天レシピデータセット
卵料理カテゴリの5647件で学習

## 学習結果
stage2は画像サイズや分散表現等の改善の余地あり\
![Stage-1](https://user-images.githubusercontent.com/38776830/51962656-3ee5e100-24a4-11e9-9515-5d4bdbe6a5a4.png)
stage1 600epoch

![Stage-2](https://user-images.githubusercontent.com/38776830/51962706-6e94e900-24a4-11e9-8cfb-7eac521382fa.png)
stage2 600epoch

## License
MIT\
こちらを元に
https://github.com/hanzhanggit/StackGAN-Pytorch