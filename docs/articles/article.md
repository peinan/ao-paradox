---
title: "アンドロイドは「青信号」の夢を見るか？"
emoji: "🚥"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Tech", "AI", "生成AI", "機械学習", "VLM"]
published: false
---

:::message
これは [CyberAgent AI Lab アドベントカレンダー](https://adventar.org/calendars/11573) 20 日目の記事です。
:::

こんにちは、AI Lab 自然言語処理 (NLP) チームの張です。
突然ですが、以下の写真をご覧ください。光っている信号の色は何色に見えますか？

![](https://images.unsplash.com/photo-1584649525122-8d6895492a5d?q=80&w=3570&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

緑色に見えますが、この信号のことを日本語で一般的に「青信号」って呼びますよね？

このように、日本では青信号や青菜、青リンゴ、青汁など、緑色に見えるものでも「青」と呼ぶ文化があります。
いろんな歴史的な経緯があるかと思いますが、「色」という概念は文化や言語を通じて形成されてきたもの[^1]であり、逆に言えば言葉は色の意味と、その色の範囲を決定づける上で重要な役割を果たしています。[^2]

ではこの言語的な色の広がりは、AI の視覚認識にも影響を与えているのでしょうか？
もっと詳しくいうと、異なる言語で学習された AI モデルは、異なる色の境界線を持ち、同じ色や写真を見せても、異なる色として認識するのでしょうか？もしそうなら、そこにどのような違いがあり、何が影響を与えているのでしょうか？
このような「文化的な色の境界線」を探るため、この記事では Vision-Language モデル (VLM) を使い、英語と日本語における「青」と「緑」を例に検証していきます。

## 検証方法

![](/images/do-androids-dream-of-blue-traffic-light/overview.jpg)
*検証方法の概要 (関係ないけど Nano Banana Pro すごい)*

基本的に、いろんなモデルに対して、いろんな画像と、異なる言語の「青」と「緑」を問うプロンプトペアを入力し、埋め込みベクトルと画像埋め込みのコサイン類似度を計算したのち、 Softmax 関数で確率化した結果を比較していきます。

画像とプロンプトは以下の 2 種類です。

- **実験1**: 色相を変化させた単色のカラーパッチと英語 (`en`) もしくは日本語 (`ja`) のプロンプトペア `["green color", "blue color"]`, `["緑色", "青色"]` を各モデルに入力し、出力から「青」の確率を計算します
- **実験2**: 色相を変化させた実際の写真と英語 (`en`) もしくは日本語 (`ja`) のプロンプトペア `the color of the {object} is green/blue`, `この{object}の色は緑色/青色` を各モデルに入力し、出力から「青」の確率を計算します
  - `{object}` は変数で、青信号、青菜、青リンゴ、青汁などが入ります

使用するモデルは以下の 3 種類です。

- 英語モデル: [`laion/CLIP-ViT-L-14-laion2B-s32B-b82K`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) (本記事中の ID: `laion`)
- 日本語モデル: [`llm-jp/llm-jp-clip-vit-large-patch14`](https://huggingface.co/llm-jp/llm-jp-clip-vit-large-patch14) (本記事中の ID: `llmjp`)
- 多言語モデル: [`timm/ViT-SO400M-14-SigLIP-384`](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) (本記事中の ID: `siglip`)

なお、詳しいモデルの選定方法や基準については余談セクションを参照してください。

:::details 前提: CLIP モデルとは
CLIP (Contrastive Language-Image Pre-training) は、OpenAI が 2021年に発表した、画像とテキストを同じ「意味空間」に結びつけたモデルで、現在の生成 AI や画像検索技術の基盤となっている重要な技術です。[^7]

いろんな解説記事がでているのでここでは詳しく触れませんが、「写真を見て、それが何の説明なのかを理解できる」、あるいは「説明文を読んで、それに合う写真を選べる」ように訓練されており、従来の画像認識で「猫」「犬」と決まったラベルを当てるだけに対し、CLIP では「夕暮れ時に海岸で走っているゴールデンレトリバー」といった、より複雑で自由な文章と画像を関連付けることができます。
:::

## 実験1: 単色カラーパッチ

まずは物体認識などのバイアスを排除した、色そのものに対する認識能力を検証するため、単色のカラーパッチをそれぞれのモデルに入力して、その反応を分析したいと思います。

### データ作成

HSV 色空間[^8]において、色相角 (H) を 60°（黄緑）から 300°（青紫）まで 1° 刻みで変化させた単色のカラーパッチを作成します。
明度 (V) は 100 で固定し、彩度 (S) は 10 から 100 まで 10 刻みにしています。各パッチは使用モデルに合わせて 224 × 224 ピクセルです。

![](/images/do-androids-dream-of-blue-traffic-light/task1_data_color_patches.png)
*作成されたカラーパッチの例*

![](/images/do-androids-dream-of-blue-traffic-light/task1_data_color_wheel.png)
*全カラーパッチをプロットしたカラーホイール*

### 実験結果

彩度 (S) が 100 の場合、以下のような結果になりました。

![](/images/do-androids-dream-of-blue-traffic-light/task1_result_s100_laion_en_llmjp_ja.png)
*色相角に対する「青」の確率曲線（彩度 100、モデル: `laion_en`, `llmjp_ja`）*

英語モデル (`laion_en`) は 174° 付近で「緑」から「青」に切り替わり、日本語モデル (`llmjp_ja`) は 177° 付近で「緑」から「青」に切り替わっており、右端では日本語モデルの方が早く「青」の確率が下がっていることが見て取れます。
これはつまり英語圏の方が日本語圏よりも「青」という言葉が指す範囲が広いということになり、当初建てていた「日本語の方が青が示す範囲は英語よりも広い」という仮説が間違っていたことになります。正直驚きました。

ちなみに 174° と 177° のカラーパッチは以下になります。

![](/images/do-androids-dream-of-blue-traffic-light/task1_h174_h177.png)

みなさんの目にはどう見えますか？

次に多言語モデル (`siglip_en`, `siglip_ja`) の方も見てみましょう。

![](/images/do-androids-dream-of-blue-traffic-light/task1_result_s100_siglip_en_siglip_ja.png)
*色相角に対する「青」の確率曲線（彩度 100、モデル: siglip_en, siglip_ja）*

差が小さいものの、同じ傾向が見られました。

差が小さくなっている原因として、多言語モデルの場合、使用される学習データの割合が日本語の方が圧倒的に低く、英語圏の感覚に偏っているからかもしれませんね。

## 実験2: 実際の写真

実験1では純粋な色に対する反応を検証しましたが、物体認識が色彩判定に与えるバイアスも無視できないファクターです。そこで、4 枚の写真を用意し、それぞれの緑色部分に対して色相回転を適用したのちに同様な方法で「青」の確率を計算します。

### データ作成

写真は以下の 4 種類です。

- 青信号
- 青リンゴ
- 青菜
- 青汁

各トピックの写真を人力で収集し、それぞれに対して実験1と同様に色相角を 60° から 300° まで 1° 刻みで回転させた画像（計 240 枚）を作成しました。

![](/images/do-androids-dream-of-blue-traffic-light/task2_data_overview.png)
*人力で収集した写真 (クレジットは脚注[^9]に記載)*

![](/images/do-androids-dream-of-blue-traffic-light/task2_data_images.png)
*色相角を変化させた写真の一部*

また、使用するプロンプトは以下になります。

| 物体 | 英語プロンプト | 日本語プロンプト |
|------|--------------|----------------|
| 青信号 | the color of illuminated circle light is green/blue | 光っている丸いライトの色は緑色/青色 |
| 青リンゴ | the color of the apple is green/blue | このリンゴの色は緑色/青色 |
| 青菜 | the color of the vegetables is green/blue | この野菜の色は緑色/青色 |
| 青汁 | the color of the liquid in the glass is green/blue | グラスに入った汁の色は緑色/青色 |

### 実験結果

実験1と同様に、色相角を変化させた画像に対して各モデルの「青」確率を計算しました。

![](/images/do-androids-dream-of-blue-traffic-light/task2_result_traffic_light_laion_en_llmjp_ja.png)
*色相角に対する「青」の確率曲線（写真: 青信号、モデル: `laion_en`, `llmjp_ja`）*

大変興味深いことに、信号機の写真の場合だと実験1とは逆の結果を示しており、日本語モデルの方が英語モデルより大幅に早く「青」と判定しているし（`llmjp_ja`: 178°、`laion_en`: 187°）、青がカバーしている範囲も日本語モデルの方が広いことが見て取れます。

ちなみに 178° と 187° の写真は以下になります。

![](/images/do-androids-dream-of-blue-traffic-light/task2_traffic_light_h178_h187.png)

みなさんの目にはどう見えますか？

他の写真についても見てみましょう。

![](/images/do-androids-dream-of-blue-traffic-light/task2_result_apple_laion_en_llmjp_ja.png)
*色相角に対する「青」の確率曲線（写真: 青リンゴ、モデル: `laion_en`, `llmjp_ja`）*

![](/images/do-androids-dream-of-blue-traffic-light/task2_result_green_vegetable_laion_en_llmjp_ja.png)
*色相角に対する「青」の確率曲線（写真: 青菜、モデル: `laion_en`, `llmjp_ja`）*

![](/images/do-androids-dream-of-blue-traffic-light/task2_result_aojiru_laion_en_llmjp_ja.png)
*色相角に対する「青」の確率曲線（写真: 青汁、モデル: `laion_en`, `llmjp_ja`）*

青リンゴと青菜は英語モデルも日本語モデルもほぼ同じタイミングで「青」と判定している（どちらも 168° 付近）一方で、青汁は日本語モデルの方がかなり早く「青」と判定していることが見て取れます（`llmjp_ja`: 166°、`laion_en`: 179°）。

ちなみにそれぞれの境界線付近の写真は以下になります。

![](/images/do-androids-dream-of-blue-traffic-light/task2_apple_green_vegetable_aojiru.png)

実験1では「英語モデルの方が青の範囲が広い」という意外な結果が出ましたが、実験2では一転して日本語モデルが「緑色の物体」を「青」と呼び始めるタイミングが早まるという現象が確認されました。

面白いのは、「青リンゴ」や「青菜」では英語モデルとの差が小さかった点です。 これは、英語でも "Green Apple" や "Green Vegetables" といった表現が一般的であり、両モデルにとって「この物体は緑系である」という認識が一致しやすいためだと推測されます。

一方で「青汁」に関しては、英語圏にそのまま対応する概念が乏しく、英語モデルが純粋に「液体（物理的な色）」として判定しているのに対し、日本語モデルは「青汁（概念としての青）」として判定しているため、大きな乖離が生まれたのではないでしょうか。

つまり、日本語の「青」は物理的な範囲こそ英語より狭い可能性があるものの、文化的なトリガーによってその境界線を柔軟に（あるいは強引に）押し広げる特性を持っていると言えそうです。

## まとめ

本記事では、VLM（Vision-Language Model）を用いて、日本語と英語における「青」と「緑」の境界線がどのように異なるのかを検証しました。

今回の検証は、AI が「色」をどう認識するかという一つの側面に過ぎませんが、言語と視覚認識の関係を探る上で興味深い結果が得られました。AI が「青信号」の夢を見るかどうかは、まだ分かりませんが、少なくとも日本語で学習した AI は、英語で学習した AI とは異なる「青」の境界線を持っているようです。

現在の VLM の多くは英語圏の基準、もしくは英語データが支配的で作られていますが、今回の検証が示す通り、言語背景が異なれば「正解」とされる色の境界線すら異なります。真にローカライズされた、あるいは文化的多様性を理解するAIを構築するためには、こうした「言葉がもたらす認識のバイアス」を深く理解することが不可欠です。

今回の検証はあくまで一例に過ぎませんが、色の境界線という小さな隙間から、AIと文化の深い関係性を垣間見ることができました。皆さんの目には、今日の信号は何色に映っているでしょうか？

## 余談

本編には書ききれなかった余談を少し書いておきます。

### モデル選定

公平な比較を行うため、できるだけ同じ条件（アーキテクチャやパッチサイズ、パラメータ数など）になるように選定をしていきます。

#### 英語モデル
英語モデルは LAION の [`laion/CLIP-ViT-L-14-laion2B-s32B-b82K`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) を使用します。
これは LAION-5B データセットの英語サブセットである LAION-2B を用いて学習された標準的な CLIP モデルで、画像エンコーダとテキストエンコーダの両方を含む Vision Transformer Large (ViT-L/14) アーキテクチャを採用しています。
OpenCLIP フレームワークを用いて大規模分散訓練により学習されたパラメータ数 428M、パッチサイズ 14 のモデルです。

#### 日本語モデル
日本語モデルは llm-jp の [`llm-jp/llm-jp-clip-vit-large-patch14`](https://huggingface.co/llm-jp/llm-jp-clip-vit-large-patch14) を使用します。
これは画像エンコーダ、テキストエンコーダの両方を、日本語データ [^4] でフルスクラッチで学習させたモデルです。
英語モデルと同じ ViT-L/14 アーキテクチャを採用しており、パラメータ数 467M とほぼ同等のモデルになります。

#### 多言語モデル
多言語モデルは Google 発の SigLIP モデルの timm 版 ([`timm/ViT-SO400M-14-SigLIP-384`](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384)) を採用します。[^5]
SigLIP モデルは ViT-L/14 とは少し異なるアーキテクチャを採用していますが、パッチサイズが 14 でパラメータ数が 400M 級のモデルで、事前検証で日本語性能が高かったため採用しました。[^6]

### 彩度による VLM の反応の違い

実験1では、彩度を 100 に固定していましたが、彩度を 10 から 100 まで 10 刻みで変化させた場合の結果は以下になりました。

![](/images/do-androids-dream-of-blue-traffic-light/task1_result_laion_en.png)
*色相角に対する「青」の確率曲線（彩度 10-100、モデル: `laion_en`）*

彩度が低いほど「青」と判定されやすくなっていることがわかります。英語圏の学習データ（LAION）において、「Blue」は鮮やかな青色だけでなく、曇りの空、遠くの海、影、夜景など、彩度が低くグレーに近い色に対しても広く使われる傾向がある一方、「Green」は植物など、比較的高彩度で鮮やかな対象物に使われることが多いです。つまり、モデルは「色彩情報が乏しく、色がはっきりしない寒色系の色」に対しては、「Green」よりも「Blue」である確率が高いと学習している可能性が高いと推測されます。

また、モデルごとの彩度による青の境界線もプロットしてみました。

![](/images/do-androids-dream-of-blue-traffic-light/task1_result_blue_boundaries_for_s.png)

ここでもやはり英語モデルが緑側に寄っており、日本語モデルは青側に寄っていることがわかり、実験1の結果を裏付ける結果になっていますね。

[^1]: この考え方は言語学や認知心理学において「言語相対性（サピア＝ウォーフ仮説）」として知られています。
[^2]: ナミビアのヒンバ族は逆に「青」を指す言葉がなく、全て緑色の仲間として扱います。Roberson らの研究 [^3] では、彼らは西洋人が一瞬で見分けられる「青と緑の違い」を区別するのが難しく、逆に西洋人が同じに見える「複数の緑色の違い」を瞬時に見分けることが可能だと報告しています。
[^3]: D. Roberson, J. Davidoff, I. R.L. Davies, L. R. Shapiro, [Color categories: Evidence for the cultural relativity hypothesis](https://www.sciencedirect.com/science/article/abs/pii/S0010028504000763), Cognitive Psychology, 2005.
[^4]: 便宜上日本語データと書いていますが、より正確には LAION-5B の英語サブセットを日本語に翻訳したものなので、少し注意が必要です。
[^5]: 実験の便宜上、インターフェイスを OpenCLIP に揃えたかったので対応している [`timm/ViT-SO400M-14-SigLIP-384`](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) を使用していますが、どうやらトークナイザーがうまく学習されておらず日本語が正しく扱えなかったので、そこだけ Google のオリジナルの [`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384) を使用しています。
[^6]: 事前に他にも [`sentence-transformers/clip-ViT-B-32-multilingual-v1`](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) や [`M-CLIP/XLM-Roberta-Large-Vit-L-14`](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14) に対して犬や猫を認識するタスクで試したのですが、総じて日本語性能が低かったため採用しませんでした。
[^7]: A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML, 2021.
[^8]: 本当は HSV 色空間ではなく、人間の色覚により近い CIELAB 色空間を使うべきですが、今回はわかりやすさ重視で HSV 色空間を使っています。
[^9]: 青信号: [Unsplash](), 青菜: [Unsplash](), 青リンゴ: [Unsplash](), 青汁: [Unsplash]()
