# **境界の色彩論：視覚言語モデルにおける「青信号（Grue）」の現象学的・計量言語学的分析**

## **概要**

近年の深層学習の進展、とりわけTransformerアーキテクチャの視覚領域への拡張は、Vision-Language Models（VLMs）という新たなパラダイムを確立した。CLIPやLLaVAに代表されるこれらのモデルは、画像とテキストを共通の埋め込み空間へ射影することで、高度なマルチモーダル推論を実現している。しかし、これらのモデルは学習データに含まれる言語的・文化的バイアスを色濃く反映することが知られており、物理的な視覚情報と言語的な意味ラベルの間に乖離が生じる場合がある。

本報告書は、日本語特有の色彩概念である「青信号（Ao-Shingo）」—物理的には緑色（Green）のスペクトルを持ちながら、言語的には「青（Blue）」とカテゴライズされる現象—を題材に、AIの視覚認識におけるサピア・ウォーフ仮説（言語相対性論）の影響を網羅的に調査・考察するものである。ACL、EMNLP、CVPR、ICCV等の主要会議における先行研究を精査し、RGB色空間の単純な勾配分析を超えた、CIELAB/IPT色空間を用いた心理物理学的アプローチを提唱する。また、英語圏のデータで訓練されたモデルと、RinnaやStability AIによる日本語特化型モデルの挙動を比較することで、自動運転などのセーフティクリティカルな応用における「文化的幻覚（Cultural Hallucination）」のリスクを明らかにし、将来の研究の方向性を提示する。

## ---

**1\. 序論：視覚と言語の不整合性**

### **1.1 マルチモーダルAIの台頭と課題**

人工知能研究において、画像認識（Computer Vision: CV）と自然言語処理（Natural Language Processing: NLP）は長らく独立した領域として発展してきた。しかし、BERT 1 やGPT 2 の登場による言語モデルの飛躍的な性能向上、およびVision Transformer (ViT) 1 による画像処理の革新は、両者を統合するVision-Language Models (VLMs) の開発を加速させた。CLIP (Contrastive Language-Image Pre-training) 3 は、インターネット上から収集された4億組の画像とテキストのペアを用いた対照学習により、ゼロショットでの画像分類や検索において驚異的な性能を示した。

しかし、この「驚異的な性能」は、学習データが内包する文化的・言語的偏向（バイアス）の上に成り立っている。VLMsは物理的な世界を直接「見て」いるのではなく、テキストというフィルターを通して世界を解釈しているに過ぎない。したがって、言語によって世界の分節化（Categorization）が異なる概念—特に色彩—においては、モデルの認識と物理的実態との間に齟齬が生じる可能性がある。

### **1.2 「青信号」のパラドックス**

日本語における「青（Ao）」は、歴史的に「緑（Midori）」を含む広いスペクトル（Grue: Green \+ Blue）を指す語彙であった 4。現代日本語では「緑」が定着しているものの、交通信号機の「進め」の灯火、野菜（青菜）、植物（青葉）など、特定の文脈においては依然として緑色の物体が「青」と呼称される。

提案者のアイデアである「『青信号』は、RGBスペクトルのどこまで緑になれば『緑信号』と呼ばれるのか？」という問いは、単なるトリビア的な興味にとどまらず、AIの認識論における核心的な課題を突いている。すなわち、\*\*「AIは画素（物理）を見ているのか、それともラベル（意味）を見ているのか」\*\*という問題である。

もしAIが物理的な波長（約530nmの緑）を認識しているならば、それは「Green/Midori」と分類されるべきである。しかし、もしAIが言語的な共起関係（文脈）を重視するならば、交通信号機の形状をした物体における緑色は「Blue/Ao」と分類される可能性がある。この現象を解明することは、自動運転車が異文化圏（例：日本）の交通システムに適応できるか、あるいは「青い信号」という指示を誤解釈して事故を引き起こさないかという、極めて実用的な安全性評価に直結する。

## ---

**2\. 理論的背景と先行研究の体系化**

本章では、ACLやCVPRなどの主要会議で議論されている理論的枠組みと、本研究課題に関連する既存の知見を整理する。

### **2.1 言語相対性論（サピア・ウォーフ仮説）とAI**

言語相対性論（Linguistic Relativity）、通称サピア・ウォーフ仮説は、「話者が使用する言語の構造が、その話者の世界観や認識に影響を与える」とする考え方である 3。

* **強い仮説（決定論）:** 言語が思考を決定する。  
* **弱い仮説:** 言語は思考や知覚に影響を与える（バイアスをかける）。

認知科学の分野では、強い仮説は概ね否定されているものの、色知覚などの特定の領域においては、言語ラベルが知覚的弁別に影響を与える（Category perception）という証拠が多数報告されている 6。

AIにおける再来:  
興味深いことに、近年の大規模言語モデル（LLM）やVLMの研究において、この仮説が「AIの認知構造」として復活しつつある。VLMは言語データ（テキスト）による教師あり学習、あるいは弱教師あり学習に依存しているため、その「視覚野」の形成は「言語野」の語彙構造に強く拘束される。

* 3 (Ottenheimer, 2009; Recent works): MLLM（Multimodal Large Language Models）は、学習した言語情報に基づいて推論を行うため、言語的決定論の「強いバージョン」に近い制約を受ける可能性があると指摘されている。  
* 5 (OpenPaws): 翻訳によってプロンプトを変えると、AIの出力が変化する現象は、言語がAIの「思考」に干渉している証左である。

### **2.2 「Grue」色彩カテゴリーとBerlin & Kayの進化論**

色彩人類学における記念碑的研究であるBerlin & Kay (1969) は、言語における基本色彩語（Basic Color Terms）の進化には普遍的な順序があるとし、多くの言語が「青」と「緑」を区別しない「Grue（Green \+ Blue）」の段階を経るとした 8。

* **World Color Survey (WCS):** 110の未文字化言語における色彩命名データを収集したWCS 10 は、色彩カテゴリーの境界が文化によって流動的であることを示している。  
* **Grueの重心:** 「Grue」を持つ言語話者にとって、そのカテゴリーの焦点（Focal color）は、純粋な青にある場合もあれば、緑にある場合もある 12。

AI研究におけるGrue:  
最近の研究 9 "Foundation Model ColorMap" (CHI 2025 / arXiv) は、VLMがこれらの色彩知識をどのようにコード化しているかを可視化するフレームワークを提案している。この研究では、モデルがWCSのデータセットをどのようにクラスタリングするかを分析し、AIが人間と同様の「色彩の普遍性と多様性」を獲得しているか検証している。しかし、特定の文化的アーティファクト（信号機）における文脈依存の色彩変容（青信号）に焦点を当てた研究は依然として希薄である。

### **2.3 日本の信号機における「青」の特殊性**

日本の交通信号機における「青」は、二重の歪みを抱えている。

1. **言語的歪み:** 緑色の光を「青」と呼ぶ 4。  
2. **物理的歪み:** 国際的な「緑」の基準を満たしつつ、日本人の「青」という語感に近づけるため、日本の信号機の緑色は、世界標準よりもわずかに青みがかった（青緑色、blue-green）波長（約500-505nm付近）が採用される傾向にある 4。

これはAIにとって極めて難解な「敵対的（Adversarial）」な状況を作り出す。

* **英語圏モデル:** 「Green」と学習した領域（530nm）と「Blue」と学習した領域（470nm）の中間にある日本の信号機を、どちらに分類するか？  
* **自動運転の現場:** TeslaのAutopilotが日本の信号機を認識する際、UI上で青く表示されたり、あるいは認識率が低下したりする事例が報告されている 14。これは、学習データの分布（US中心）と推論環境（日本）の不一致（Distribution Shift）によるものである。

## ---

**3\. 既存研究の網羅的調査とギャップ分析**

### **3.1 VLMの色彩能力評価 (Benchmarks & Probing)**

AIが色をどの程度理解しているかに関する研究は、ACLやCVPRで散見される。

* Rainbow Benchmark 16: Flickr30kやMS COCOなどのデータセットを用い、色に関する記述を置換した際のモデルの正誤判定能力を測定するベンチマーク。  
  * *限界:* 基本的に英語データセットであり、西洋的な色彩カテゴリー（11基本色）を前提としている。  
* Visual Probing & ColorMap 9: モデルの内部表現（Embedding）を解析し、色がどのようにマッピングされているかを可視化する試み。  
  * *知見:* CLIP等のモデルは、色相環上の隣接色（青と緑など）の区別に苦戦する場合がある 18。また、物体と色の結びつき（Bias）が強く、「青いバナナ」のような反事実的な組み合わせの認識精度が落ちる。

### **3.2 文化的バイアスの検証 (Cultural Bias in VLMs)**

VLMが西洋中心（Western-centric）であるという批判は、近年急速に高まっている 19。

* 19 (Western Bias in VLMs): 西洋の文化圏の画像や概念に対しては高い精度を示すが、非西洋圏のデータに対しては精度が落ちる、あるいは西洋的な解釈を押し付ける傾向がある。  
* 21 (Cultural Bias Visualization): 性別や地域によるバイアスの交差性（Intersectionality）を可視化しているが、色彩語彙におけるバイアス（Grueなど）への言及は限定的である。

### **3.3 日本語特化型VLMの開発**

多言語モデル（Multilingual CLIP）の限界を克服するため、日本語に特化したモデルが開発されている。

* Rinna Japanese CLIP 22: 日本語のキャプション付き画像（CC12Mのフィルタリング版など）で学習。トークナイザーが日本語に最適化されている。  
* Japanese Stable CLIP 23: Stability AIによるモデル。より大規模なデータとモデルサイズ（ViT-L/16）で、日本の文化的アイテム（鳥居、着物、日本の食事など）の認識能力が向上している 23。

**【Deep Dive点 1】:** 既存研究では、これらの日本語モデルが「着物」や「寿司」といった\*\*名詞（オブジェクト）\*\*の認識において優れていることは示されているが、**形容詞（色彩）の境界線**、特に「青」と「緑」のスペクトル境界が英語モデルとどのように異なっているかを定量的に比較した研究は皆無に近い。

### **3.4 自動運転と信号機認識 (Traffic Light Recognition)**

自動運転技術の文脈では、信号機認識（TLR）はYOLOなどの物体検出モデルで行われるのが一般的である 24。

* **データセットの偏り:** 既存の主要な信号機データセット（Bosch Small Traffic Lights Dataset, DriveUなど）は欧米の都市で収集されており、日本の信号機を含んでいないことが多い 26。  
* **分類クラス:** 多くのデータセットは {Red, Yellow, Green} の3クラス分類であり、「Blue」というクラスは存在しない。したがって、モデルは日本の青信号を無理やり「Green」に分類するか、あるいは背景（Noise）として棄却するリスクがある 27。

## ---

**4\. 提案手法：論文化に向けたDeep Diveと実験設計**

ユーザーのアイデアをトップカンファレンス（ACL/CVPR）レベルの論文に昇華させるためには、単なる「可視化」を超えた、科学的に厳密な実験設計が必要である。以下に、既存研究の欠落を埋めるための具体的なメソドロジーを提案する。

### **4.1 色空間の再考：RGBからの脱却とCIELAB/IPTの導入**

ユーザーのアイデアにある「RGBスペクトル」での実験は、学術的には不十分である。RGB色空間は知覚的に均等（Perceptually Uniform）ではないため、RGB値の線形補間が人間の知覚上の線形変化と一致しないからである。

推奨アプローチ:  
実験には CIELAB ($L^\*a^\*b^\*$) または IPT 色空間を使用すべきである 28。

* **理由:** 青（Blue）から緑（Green）への変化を検証する際、CIELABは色相の直線性に課題があることが知られている（Blueの領域で色相が曲がる現象）。IPT色空間は、色相の恒常性（Constant Hue）を保つように設計されており、青-緑間のグラデーション評価に最適である 30。  
* **実験刺激の生成:** IPT色空間上で、Hue角を青から緑へ一定刻みで変化させつつ、明度（Lightness）と彩度（Chroma）を固定した「制御された刺激画像」を生成する。

### **4.2 比較対象モデルと実験設定**

| モデルカテゴリ | 代表的モデル | 役割 |
| :---- | :---- | :---- |
| **English Native** | OpenAI CLIP (ViT-B/16, L/14) | 西洋的認知のベースライン。英語キャプションで学習。 |
| **Multilingual** | LAION-5B (OpenCLIP) | 多言語データによる「普遍性」の検証。データ量によるバイアス希釈効果の確認。 |
| **Japanese Native** | Rinna Japanese CLIP 22 | 日本語データのみで学習。サピア・ウォーフ効果（日本語構造の反映）の検証主力。 |
| **Japanese Native** | Stability AI Japanese Stable CLIP 23 | 高性能な日本語モデルでの再現性確認。 |

### **4.3 心理測定関数（Psychometric Function）の算出**

本研究のアプローチは、AIを被験者と見立てた心理物理学実験（AI Psychophysics）として定式化する。

1. **プロンプトエンジニアリング:**  
   * 単一のプロンプト（例：「青い信号」）ではノイズが大きいため、**Prompt Ensembling** 16 を採用する。  
   * 英語: "A photo of a {color} traffic light", "A {color} signal", "The traffic light is {color}"  
   * 日本語: 「{色}の信号機の写真」「{色}信号」「信号機は{色}です」  
   * ここでの {色} には、対立概念としての「Blue/Green」および「青/緑」を挿入する。  
2. **ゼロショット分類確率の算出:**  
   * 生成した色相グラデーション画像 $x\_i$ に対し、各モデルのテキストエンコーダーでエンコードされた「青（Blue）」と「緑（Green）」の埋め込みベクトルとのコサイン類似度を計算し、Softmax関数を用いて確率 $P(\\text{Blue}|x\_i)$ と $P(\\text{Green}|x\_i)$ を算出する。  
3. **主観的等価点（PSE: Point of Subjective Equality）の特定:**  
   * $P(\\text{Blue}) \= P(\\text{Green}) \= 0.5$ となる色相角（Hue Angle）を特定する。  
   * **仮説:** 日本語モデルのPSEは、英語モデルのPSEよりも「緑」側の波長にシフトしているはずである（すなわち、物理的に緑であっても「青」と判定する確率が高い）。

### **4.4 文脈依存性の検証（Contextual Dependency）**

「青」というラベルの使用は文脈に依存する。単なる色パッチ（Color Patch）と、信号機の画像（Object Context）とで、境界線がどう移動するかを比較する。

* **条件A（脱文脈化）:** 単色の正方形パッチ。  
* **条件B（文脈化）:** 信号機の形状をした画像（生成モデルやスタイル変換で色相のみ操作）。  
* **予想される結果:** 日本語モデルにおいてのみ、条件B（信号機）の場合に「青」の領域が緑色側へ侵食する現象（Cognitive Penetration）が観測されると予測される。これは「信号機」というオブジェクト認識が、色彩判定の事前分布を歪めることを示唆する。

## ---

**5\. 予想される結果と考察：Deep Dive**

### **5.1 「Grue」のスペクトル境界の可視化**

先行研究 4 から、英語モデルにおけるBlue/Greenの境界は、波長495nm（シアン）付近に明確に現れると予想される。一方、日本語モデルでは、この境界が500nm〜510nm、あるいはそれ以上まで「Blue（青）」側に吸収される可能性がある。  
特に注目すべきは、「緑（Midori）」という単語が存在するにもかかわらず、「青（Ao）」が勝つ領域の特定である。現代日本語モデルであっても、学習データ内の「青信号」という共起頻度が圧倒的であれば、視覚的な緑特徴量よりも、言語的な確率共起が優先される現象が確認できるだろう。これは、VLMsが視覚的グラウンディング（Visual Grounding）よりも、分布的意味論（Distributional Semantics）に強く依存していることを示唆する。

### **5.2 翻訳レイヤーによる「二重の歪み」**

多言語対応を謳うサービス（例：GPT-4VのAPI利用時など）において、内部で「日本語プロンプト→英語への翻訳→画像認識」というプロセスが走っている場合、致命的なエラーが発生する可能性がある 5。

* ユーザー入力：「青信号を見て進んだ」  
* 機械翻訳：「I saw the **blue** light and proceeded.」  
* 画像認識部：画像は「Green」。しかしテキストは「Blue」。  
* 結果：モデルは画像の「Green」とテキストの「Blue」の不一致（Hallucination）を検知し、誤った修正を行うか、認識そのものを失敗する。  
  本研究により、日本語ネイティブモデルの優位性が、単なる語彙数だけでなく、こうした「概念の直接的マッピング」にあることを論証できる。

### **5.3 自動運転システムへの警鐘**

TeslaやWaymoなどの自動運転システムが、基本色カテゴリー（Basic Color Categories）として英語の枠組み（11色）を採用している場合、日本の「青信号」はエッジケースとなる 15。  
「青みがかった緑」は、国際標準では「Green」の許容範囲内かもしれないが、モデルが学習データ（カリフォルニアの鮮やかな緑）に過学習している場合、日本の信号機を「Unknown」や「Blue（=Meaningless/Decoration）」と分類するリスクがある。本研究の結果は、自動運転AIのローカライゼーションにおいて、単なる道路標識の再学習だけでなく、\*\*色彩認識のキャリブレーション（再調整）\*\*が不可欠であることを示唆するデータとなる。

## ---

**6\. Future Work（今後の展望）**

本研究を足がかりに、以下のような発展的研究が可能である。

### **6.1 Dynamic Cultural Adapters (文化適応アダプタ)**

モデル全体を再学習するのではなく、LoRA (Low-Rank Adaptation) 技術を用いて、特定の文化圏（ロケール）に合わせて色彩境界を動的にシフトさせる「Cultural Adapter」の開発 3。

* **Japan-Color-LoRA:** これを適用すると、信号機や野菜を見たときの「青」の活性化領域が広がる。  
* Western-Color-LoRA: これを適用すると、厳密なスペクトル区分に戻る。  
  これにより、単一の基盤モデルで多文化に対応可能な「Culturally Aware AI」の実現に寄与する。

### **6.2 空間的・感情的相対性の検証**

色彩だけでなく、空間表現（Egocentric vs Geocentric）や感情分析における色の意味（白：西洋では純潔、東洋の一部では喪）など、他のモダリティにおけるサピア・ウォーフ効果の検証へと拡張する 21。特に、空間認識における「前後左右」の概念と、言語による空間記述の相関をVLMで調査することは、ロボティクス分野で重要となる。

### **6.3 「Grue Test」ベンチマークの構築**

本研究のメソドロジーを標準化し、VLMsの文化受容性を測るベンチマークテスト「The Grue Test」として公開する。

* ISO標準に準拠した色彩パッチと、文化依存性の高いオブジェクト画像のペアセット。  
* 多言語での評価スコア（Cultural Alignment Score）の算出。  
  これは、Hugging Faceなどのリーダーボードにおいて、モデルの「多言語性能」を測る新たな指標となり得る。

## ---

**7\. 結論**

本報告書は、ユーザーが提案した「青信号の色彩境界」に関するアイデアを、最新の視覚言語モデル研究の文脈に位置づけ、その学術的・社会的意義を論じた。既存研究の調査から、現在のVLMsは言語的な学習データに依存した知覚バイアス（サピア・ウォーフ効果）を内包しており、特に日本語の「青」のようなGrueカテゴリーにおいて、英語圏モデルとの間に看過できない認知のズレが生じている可能性が高いことが明らかになった。

提案されたアプローチ—日本語モデルと英語モデルの色彩境界の比較—は、単なる興味本位の実験を超え、AIの**公平性（Fairness）**、**安全性（Safety）**、そして\*\*包括性（Inclusivity）\*\*を検証するための重要な試金石となる。CIELAB/IPT色空間を用いた厳密な実験設計と、文化特化型モデル（Rinna/Stability）の比較分析を通じて、本研究は「真にグローバルなAI」の実現に向けた重要な知見を提供し、ACLやCVPRといったトップカンファレンスにおいて、計算言語学とコンピュータビジョンの境界領域を拡張する貢献が期待できる。

## ---

**付録：主要なデータセットとツール**

以下は、本研究の遂行にあたり不可欠となるリソースの一覧である。

| リソース名 | 種別 | 概要・用途 | 参照 |
| :---- | :---- | :---- | :---- |
| **World Color Survey (WCS)** | データセット | 世界110言語の色彩命名データ。Grueの普遍的分布の参照基準（Ground Truth）として利用。 | 10 |
| **Foundation Model ColorMap** | ツール/手法 | VLMの色彩知識を抽出・可視化するフレームワーク。本研究のベースライン手法。 | 9 |
| **Rainbow Benchmark** | ベンチマーク | 色彩と言語の結びつきを評価する既存指標。比較対象として有用。 | 16 |
| **Bosch Small Traffic Lights Dataset** | データセット | 信号機画像データ。色相変換（Hue Shift）を行い、擬似的な日本信号機データを作成するために利用可能。 | 26 |
| **Rinna Japanese CLIP** | モデル | 日本語特化型VLMの代表例。Hugging Faceにて公開中。 | 22 |
| **Colour Science for Python** | ライブラリ | CIELAB/IPT色空間での正確なグラデーション生成と色差計算（Delta E）に利用。 | 28 |

以上

#### **Works cited**

1. VisMoDAl: Visual Analytics for Evaluating and Improving Corruption Robustness of Vision-Language Models \- arXiv, accessed December 11, 2025, [https://arxiv.org/html/2509.14571v1](https://arxiv.org/html/2509.14571v1)  
2. Studying second language acquisition in the age of large language models: Unlocking the mysteries of language and learning, A commentary on “Age effects in second language acquisition: Expanding the emergentist account” by Catherine L. Caldwell-Harris and Brian MacWhinney \- NIH, accessed December 11, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10927252/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10927252/)  
3. Will multimodal large language models ever achieve deep understanding of the world?, accessed December 11, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12679578/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12679578/)  
4. What a Blue Apple Can Tell You About Design | by Kelly Smith \- Medium, accessed December 11, 2025, [https://kelmarmon.medium.com/what-a-blue-apple-can-tell-you-about-design-b029595d9c9d](https://kelmarmon.medium.com/what-a-blue-apple-can-tell-you-about-design-b029595d9c9d)  
5. Literature review on developing artificial intelligence to advocate for animal rights, accessed December 11, 2025, [https://www.openpaws.ai/research-and-reports/literature-review-on-developing-artificial-intelligence-to-advocate-for-animal-rights](https://www.openpaws.ai/research-and-reports/literature-review-on-developing-artificial-intelligence-to-advocate-for-animal-rights)  
6. Can Large Language Models Affect the Way We Think? And Should We Be Afraid of Tokenized Thinking? | by Monika Zurawska | Medium, accessed December 11, 2025, [https://medium.com/@quantyment/can-large-language-models-affect-the-way-we-think-4c564b19b110](https://medium.com/@quantyment/can-large-language-models-affect-the-way-we-think-4c564b19b110)  
7. Coordinating perceptually grounded categories through language: A case study for colour \- VUB AI-lab, accessed December 11, 2025, [https://ai.vub.ac.be/sites/default/files/steels-05e.pdf](https://ai.vub.ac.be/sites/default/files/steels-05e.pdf)  
8. Understanding adversarial examples requires a theory of artefacts for deep learning, accessed December 11, 2025, [https://par.nsf.gov/servlets/purl/10267254](https://par.nsf.gov/servlets/purl/10267254)  
9. A Framework for Extracting and Visualizing the Foundation Models' Color Knowledge \- ResearchGate, accessed December 11, 2025, [https://www.researchgate.net/publication/391152583\_Foundation\_Model\_ColorMap\_A\_Framework\_for\_Extracting\_and\_Visualizing\_the\_Foundation\_Models'\_Color\_Knowledge](https://www.researchgate.net/publication/391152583_Foundation_Model_ColorMap_A_Framework_for_Extracting_and_Visualizing_the_Foundation_Models'_Color_Knowledge)  
10. Communicating artificial neural networks develop efficient color-naming systems \- PMC, accessed December 11, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8000426/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8000426/)  
11. World Color Survey color naming reveals universal motifs and their within-language diversity | PNAS, accessed December 11, 2025, [https://www.pnas.org/doi/10.1073/pnas.0910981106](https://www.pnas.org/doi/10.1073/pnas.0910981106)  
12. Color naming across languages reflects color use \- PNAS, accessed December 11, 2025, [https://www.pnas.org/doi/10.1073/pnas.1619666114](https://www.pnas.org/doi/10.1073/pnas.1619666114)  
13. Why Japan has blue traffic lights instead of green | Brian Lovin, accessed December 11, 2025, [https://brianlovin.com/hn/37982495](https://brianlovin.com/hn/37982495)  
14. Traffic light system reports the blue emergency light found by hospital and universities as a green light. If detected at the last second it will apply breaks break pretty sharply. : r/teslamotors \- Reddit, accessed December 11, 2025, [https://www.reddit.com/r/teslamotors/comments/hvmuif/traffic\_light\_system\_reports\_the\_blue\_emergency/](https://www.reddit.com/r/teslamotors/comments/hvmuif/traffic_light_system_reports_the_blue_emergency/)  
15. Tesla Autopilot now alerts drivers to green lights \- CNET, accessed December 11, 2025, [https://www.cnet.com/roadshow/news/tesla-autopilot-self-driving-green-lights/](https://www.cnet.com/roadshow/news/tesla-autopilot-self-driving-green-lights/)  
16. Rainbow A Benchmark for Systematic Testing of ... \- ACL Anthology, accessed December 11, 2025, [https://aclanthology.org/2024.eacl-long.112.pdf](https://aclanthology.org/2024.eacl-long.112.pdf)  
17. VLM's Eye Examination: Instruct and Inspect Visual Competency of Vision Language Models \- arXiv, accessed December 11, 2025, [https://arxiv.org/html/2409.14759v1](https://arxiv.org/html/2409.14759v1)  
18. The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights \- ACL Anthology, accessed December 11, 2025, [https://aclanthology.org/2025.acl-long.1102.pdf](https://aclanthology.org/2025.acl-long.1102.pdf)  
19. See It from My Perspective: How Language Affects Cultural Bias in Image Understanding, accessed December 11, 2025, [https://arxiv.org/html/2406.11665v2](https://arxiv.org/html/2406.11665v2)  
20. See It from My Perspective: Diagnosing the Western Cultural Bias of Large Vision-Language Models in Image Understanding \- arXiv, accessed December 11, 2025, [https://arxiv.org/html/2406.11665v1](https://arxiv.org/html/2406.11665v1)  
21. Cultural Bias Mitigation in Vision-Language Models for Digital Heritage Documentation: A Comparative Analysis of Debiasing Techniques, accessed December 11, 2025, [https://scipublication.com/index.php/AIMLR/article/download/120/115/264](https://scipublication.com/index.php/AIMLR/article/download/120/115/264)  
22. Mitsua/mitsua-japanese-clip-vit-b-16 \- Hugging Face, accessed December 11, 2025, [https://huggingface.co/Mitsua/mitsua-japanese-clip-vit-b-16](https://huggingface.co/Mitsua/mitsua-japanese-clip-vit-b-16)  
23. WAON: Large-Scale and High-Quality Japanese Image-Text Pair Dataset for Vision-Language Models \- arXiv, accessed December 11, 2025, [https://arxiv.org/html/2510.22276v1](https://arxiv.org/html/2510.22276v1)  
24. FlashLightNet: An End-to-End Deep Learning Framework for Real-Time Detection and Classification of Static and Flashing Traffic Light States \- PubMed Central, accessed December 11, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12568241/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12568241/)  
25. Traffic Light Recognition using Convolutional Neural Networks: A Survey \- arXiv, accessed December 11, 2025, [https://arxiv.org/pdf/2309.02158](https://arxiv.org/pdf/2309.02158)  
26. Camera-based Context-aware Traffic Light Detection for Self-Driving Vehicles, accessed December 11, 2025, [https://ml4ad.github.io/files/papers2023/Camera-based%20Context-aware%20Traffic%20Light%20Detection%20for%20Self-Driving%20Vehicles.pdf](https://ml4ad.github.io/files/papers2023/Camera-based%20Context-aware%20Traffic%20Light%20Detection%20for%20Self-Driving%20Vehicles.pdf)  
27. Current status and issues of traffic light recognition technology in Autonomous Driving System \- IEICE Transactions, accessed December 11, 2025, [https://search.ieice.org/bin/pdf\_advpub.php?category=A\&lang=E\&fname=2021WBI0002\&abst=](https://search.ieice.org/bin/pdf_advpub.php?category=A&lang=E&fname=2021WBI0002&abst)  
28. List of color spaces and their uses \- Wikipedia, accessed December 11, 2025, [https://en.wikipedia.org/wiki/List\_of\_color\_spaces\_and\_their\_uses](https://en.wikipedia.org/wiki/List_of_color_spaces_and_their_uses)  
29. Development and Testing of a Color Space (IPT) with Improved Hue Uniformity \- IS\&T | Library, accessed December 11, 2025, [https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/6/1/art00003](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/6/1/art00003)  
30. (PDF) Development and Testing of a Color Space (IPT) with Improved Hue Uniformity \- ResearchGate, accessed December 11, 2025, [https://www.researchgate.net/publication/221677980\_Development\_and\_Testing\_of\_a\_Color\_Space\_IPT\_with\_Improved\_Hue\_Uniformity](https://www.researchgate.net/publication/221677980_Development_and_Testing_of_a_Color_Space_IPT_with_Improved_Hue_Uniformity)  
31. Hue linearity of color spaces for wide color gamut and high dynamic range media, accessed December 11, 2025, [https://opg.optica.org/josaa/upcoming\_pdf.cfm?id=386515](https://opg.optica.org/josaa/upcoming_pdf.cfm?id=386515)  
32. Survey of Cultural Awareness in Language Models: Text and Beyond \- MIT Press Direct, accessed December 11, 2025, [https://direct.mit.edu/coli/article/51/3/907/130804/Survey-of-Cultural-Awareness-in-Language-Models](https://direct.mit.edu/coli/article/51/3/907/130804/Survey-of-Cultural-Awareness-in-Language-Models)