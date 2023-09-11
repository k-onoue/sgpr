# スパースガウス過程回帰

作成者 Onoue Keisuke
作成日 2022/05/14
更新日 2023/05/18

**目次**

# 0. 導入

ガウス過程とは、平均関数 $m(\bold x)$と共分散関数 $k(\bold x, \bold x’)$ によって定義される確率過程で、あらゆる $n$ 個の入力点 $\bold x_n$ の関数 $f$ による出力値 $f(\bold x_n)$ の分布が、平均 $m(\bold x)$と $K_{nn'} = k(\bold x_n, \bold x_{n'})$ を要素とする分散共分散行列 $\bold K$ の多次元ガウス分布に従うものというふうに定義されます。

直感的な説明としては、「サイコロ」を振ると 1 ~ 6 の自然数が出てくる箱と表現されるのに対して、「ガウス過程」は振ると関数 $f()$ が出てくる箱のようなものだと言われます。

このガウス過程を、分析したいデータの背後にある構造だと仮定して行う回帰の手法がガウス過程回帰です。

ガウス過程回帰の魅力は様々ですが、非線形な関係性をモデル化できる、つまりモデルとしての表現能力がとても高いことや、予測値に対しての自信の度合いを出力できることなどが挙げられます。

対して、計算量がデータ点数の３乗に比例するために、データ点数が大きくなると現実的な時間内で計算を終えるのが難しくなるといった問題もあります。

このガウス過程回帰の最大の問題点である多大な計算コストに対応するための手段の１つとして、スパース近似（補助変数法）などが提案されてきました。

この記事は、直感的にわかりやすい説明というよりは、「ガウス過程と機械学習（講談社）」を読んでいて私が躓いた箇所、式変形や論理展開などを補足する形で書いたものです。

以下を導出することをこの記事のゴールとします。

- スパースガウス過程回帰モデルの予測分布
    
$$
\begin{align*}
p(\bold f^* | \bold y)
&\approx \mathcal N (\bold f^* | (\bold K^T_{M*} \bold K^{-1}_{MM}) \widehat{\bold u}, \bold \Lambda_* + \bold K^T_{M*} \bold Q_{MM}^{-1} \bold K_{M*})
\end{align*}
$$
    
- スパースガウス過程回帰モデルの対数周辺尤度の変分下界（ELBO）
    
$$\begin{align*} \mathcal L &= \log \mathcal N (\bold y| \bold 0_N, \bold K^T_{MN} \bold K^{-1}_{MM}\bold K_{MN} + \sigma^2 \bold I) - \frac{1}{2\sigma^2}\text{Tr}(\bold \Lambda)
\end{align*}
$$
    

# 1. ガウス過程回帰モデル

通常のガウス過程回帰モデルには基本的に紹介だけに留めます。

前提は、観測誤差 $\epsilon \sim \mathcal N(0, \sigma^2)$ の観測データ $\mathcal{D} = \{ (\bold{x}_1, y_1), (\bold{x}_2, y_2) ,..., (\bold{x}_N, y_N) \}$ が与えられていて、 $y$ は平均が $0$ になるように正規化してあるとすると、入力 $\bold x_n$ と出力 $y_n$ の間に

$$
y_n = f(\bold{x}_n) + \epsilon_n, \text{ where } f \sim \text{GP}(\bold 0, k(\bold x, \bold x'))
$$

の関係があるとします。ここで $\bold f = (f(\bold x_1), \dots, f(\bold x_N))$ 、 $\bold K_{NN}$ を $K_{nn’} = k(\bold x_n, \bold x_{n’})$ を要素とする共分散行列とすると 

$$
\begin{align*}
p(\bold y | \bold f) &= \mathcal N (\bold y | \bold f, \sigma^2 \bold I) \\[10pt]
p(\bold f | \bold X) & = \mathcal N(\bold f | \bold 0, \bold K_{NN})
\end{align*}
$$

と表せます。

ここで、新たな入力 $\bold x^*$ に対して、観測値の集合 $y^*$ を考えます。新たな入力は1つでなくて良いので、それぞれ $\bold X^*$ 、 $\bold y^*$ とすれば、その予測分布は、

$$
\begin{align*}
p(\bold y^* | \bold X^*, \mathcal D) = \mathcal N  (\bold y^* | \bold K_{N*}^T (\bold \sigma^2 \bold I &+ \bold K_{NN})^{-1} \bold y, \\ &\bold K_{**} - \bold K_{N*}^T (\sigma^2 \bold I + \bold K_{NN})^{-1} \bold K_{N*})
\end{align*}
$$

で与えられます。

ハイパーパラメータの最適化については、 $\bold y$ の対数周辺尤度 $\log p(\bold y | \bold X)$ をカーネル関数のパラメータと観測ノイズについて最大化します。

まず周辺尤度 $p(\bold y| \bold X)$ は

$$
\begin{align*}
p(\bold y | \bold X)
&= \int p(\bold y, \bold f | \bold X) d\bold f \\
&= \int p(\bold y| \bold f) p(\bold f | \bold X) d \bold f \\
&= \int \mathcal N(\bold y - \bold f | \bold 0, \sigma^2 \bold I) \mathcal N(\bold f | \bold 0, \bold K_{NN}) d\bold f 
\end{align*}
$$

とかけて、これは正規分布同士の畳み込みであることがわかるので、

$$
\begin{align*}
p(\bold y| \bold X) 
&= \mathcal N (\bold y | \bold 0 + \bold 0, \sigma^2 \bold I + \bold K_{NN}) \\[10pt]
&= \mathcal N (\bold y | \bold 0, \sigma^2 \bold I + \bold K_{NN})
\end{align*}
$$

となります。

この対数をとって最適化に関係のある部分、つまりカーネル関数のハイパーパラメータと観測ノイズを含む項だけを残せばよいので

$$
\begin{align*}
\log p(\bold y | \bold X)
&= \log \mathcal N (\bold y| \bold 0, \sigma^2 \bold I + \bold K_{NN}) \\[10pt]
&= \frac{1}{(2\pi)^{N/2}} \frac{1}{|\sigma^2 \bold I + \bold K_{NN}|^{1/2}} \exp(-\frac{1}{2} \bold y^T (\sigma^2 \bold I + \bold K_{NN})^{-1} \bold y) \\[10pt]
&= - \frac{N}{2}\log(2\pi) - \frac{1}{2} \log |\sigma^2 \bold I + \bold K_{NN}| - \frac{1}{2} \bold y^T (\sigma^2 \bold I + \bold K_{NN})^{-1} \bold y \\[10pt]
&= -\log |\sigma^2 \bold I + \bold K_{NN}| - \bold y^T (\sigma^2 \bold I + \bold K_{NN})^{-1} \bold y + \text{ const} \\[10pt]
&\propto -\log |\sigma^2 \bold I + \bold K_{NN}| - \bold y^T (\sigma^2 \bold I + \bold K_{NN})^{-1} \bold y 
\end{align*}
$$

となって最適化のための目的関数が得られました。

# 2. スパースガウス過程回帰モデル

補助変数法にはいくつかの流儀があるようですが、ここでのモデルは、Fully independent training conditional (FITC) approximation と呼ばれるものです。

## 2.0 前提

基本的に、１章での通常のガウス過程回帰の設定を引き継ぎます。ここで、 $f(\cdot)$ 定義域内に、 $M < N$ 個の補助入力点 $\bold Z = (\bold z_1, \dots, \bold z_M)$ を配置し、 $\bold Z$ 上の $f(\cdot)$ による出力値 $\bold u = (f(\bold z_1), \dots, f(\bold z_M))$ を導入し、これを補助変数ベクトルと呼びます。ここで、 $p(\bold u) = \mathcal N(\bold u| \bold 0, \bold K_{MM})$ となり、 $\bold f$ と $\bold u$ の同時分布 $p(\bold f, \bold u)$ は、

$$
\begin{align*}
p(\bold f, \bold u)
&= \mathcal N \left(\begin{pmatrix}\bold f \\ \bold u \end{pmatrix}\bigg\vert \begin{pmatrix}\bold 0_N \\ \bold 0_M \end{pmatrix}, \begin{pmatrix}\bold K_{NN} & \bold K_{NM} \\ \bold K_{NM}^T & \bold K_{MM} \end{pmatrix} \right)
\end{align*}
$$

で与えられます。

補助変数法では、新しい入力点 $\bold X^*$ の予測分布に

$$
\begin{align*}
p(\bold f^* | \bold y) 
&= \int p(\bold f^*| \bold f) p(\bold f | \bold y) d\bold f \\[10pt]
&\approx \int p(\bold f^*| \bold u) p(\bold u | \bold y) d\bold u
\end{align*}
$$

という近似を行い、この近似が精度良く成り立つなら、新しい入力点の代わりに観測データの入力点 $\bold X$ を入れても

$$
\begin{align*}
p(\bold f | \bold y) 
&\approx \int p(\bold f| \bold u) p(\bold u | \bold y) d\bold u
\end{align*}
$$

が成り立ちます。ここで 観測点 $\bold y$ の生成過程は

1. $\bold u \sim \mathcal N (\bold 0_M, \bold K_{MM})$
2. $\bold f | \bold u \sim \mathcal N (\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda),$ 
$\text{ where } \bold \Lambda = \text{ diag}(\bold K_{NN} - \bold K^T_{MN} \bold K^{-1}_{MM} \bold K_{MN})$
3. $\bold y | \bold f \sim \mathcal N(\bold f, \sigma^2 \bold I_N)$ 

とします。

## 2.1 予測分布の導出

### 補助変数の事後確率 $p(\bold u| \bold y)$

次に、補助変数の事後確率 $p(\bold u | \bold y)$ を求めます。まず、ベイズの定理より

$$
\begin{align*}
p(\bold u | \bold y)
&= \frac{p(\bold y | \bold u) p(\bold u)}{p(\bold y)} \\[10pt]
\iff
\ln p(\bold u | \bold y)
&= \ln \frac{p(\bold y | \bold u) p(\bold u)}{p(\bold y)} \\[10pt]
\iff
\ln p(\bold u | \bold y)
&= \ln p(\bold y | \bold u) + \ln p(\bold u) - \ln p(\bold y)
\end{align*}
$$

となります。 $p(\bold y| \bold u)$ は畳み込みにより、

$$
\begin{align*}
p(\bold y | \bold u)
&= \int p(\bold y | \bold f) p(\bold f | \bold u) d\bold f \\[10pt]
&= \int \mathcal N (\bold y |\bold f, \sigma^2 \bold I) \mathcal N (\bold f |\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda) d\bold f \\[10pt]
&= \int \mathcal N (\bold y - \bold f |\bold 0, \sigma^2 \bold I) \mathcal N (\bold f |\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda) d\bold f \\[10pt]
&= \mathcal N (\bold y| \bold 0 + \bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \sigma^2 \bold I + \bold\Lambda) \\[10pt]
&= \mathcal N (\bold y| \bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \sigma^2 \bold I + \bold\Lambda)
\end{align*}
$$

となります。 $p(\bold y | \bold u)$ と $p(\bold u)$ を $\ln p(\bold u | \bold y)$ の左辺に代入して、 $\bold u$ に関して偏微分すると

$$
\begin{align*}
\frac{\partial \ln p(\bold u | \bold y)}{\partial \bold u}
&= \frac{\partial}{\partial \bold u} \{\ln p(\bold y | \bold u) + \ln p(\bold u) - \ln p(\bold y)\} \\[10pt]
&= \frac{\partial}{\partial \bold u} \{\ln p(\bold y | \bold u)\} + \frac{\partial}{\partial \bold u}\{ \ln p(\bold u)\} - \frac{\partial}{\partial \bold u} \{\ln p(\bold y)\} \\[10pt]
&= \frac{\partial}{\partial \bold u} \{\ln p(\bold y | \bold u)\} + \frac{\partial}{\partial \bold u}\{ \ln p(\bold u)\} 
\end{align*}
$$

で、 $\frac{\partial}{\partial \bold u} \{\ln p(\bold y | \bold u)\}$ と $\frac{\partial}{\partial \bold u}\{ \ln p(\bold u)\}$ はそれぞれ、

$$
\begin{align*}
\frac{\partial}{\partial \bold u} \{ \ln p(\bold y | \bold u) \}
&= \frac{\partial}{\partial \bold u}\{\ln \mathcal N (\bold y| \bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \sigma^2 \bold I + \bold\Lambda)\} \\[10pt]
&= \frac{\partial}{\partial \bold u} \left\{ -\frac{N}{2}\ln 2\pi - \frac{1}{2} \ln|\sigma^2\bold I + \bold \Lambda| -\frac{1}{2}(\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)^T (\bold \sigma^2 \bold I + \bold\Lambda)^{-1} (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)\right\} \\[10pt]
&= -\frac{1}{2}\frac{\partial}{\partial \bold u} \left\{ (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)^T (\bold \sigma^2 \bold I + \bold\Lambda)^{-1} (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)\right\} \\[10pt]
&= -\frac{1}{2} \frac{\partial}{\partial \bold u} \left\{ \bold y^T (\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold y - \bold y^T (\bold \sigma^2 \bold I + \bold\Lambda)^{-1}\bold K^T_{MN} \bold K^{-1}_{MM} \bold u - (\bold K^T_{MN} \bold K^{-1}_{MM} \bold u)^T(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold y + (\bold K^T_{MN} \bold K^{-1}_{MM} \bold u)^T(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \bold K^{-1}_{MM} \bold u \right\} \\[10pt]
&= -\frac{1}{2} \frac{\partial}{\partial \bold u} \left\{ - 2[\{\bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y \}^T \bold u]^T + \bold u^T \bold K^{-1}_{MM} \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \bold K^{-1}_{MM} \bold u \right\} \\[10pt]
&= -\frac{1}{2}  \left\{ - 2\bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y + 2 \bold K^{-1}_{MM} \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \bold K^{-1}_{MM} \bold u \right\} \\[10pt]
&= \bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y - \bold K^{-1}_{MM} \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \bold K^{-1}_{MM} \bold u
\end{align*}
$$

$$
\begin{align*}
\frac{\partial}{\partial \bold u} \left\{ \ln p(\bold u) \right\}
&= \frac{\partial}{\partial \bold u} \left\{ \ln \mathcal N (\bold u| \bold 0_M, \bold K_{MM}) \right\} \\[10pt]
&= \frac{\partial}{\partial \bold u} \left\{ -\frac{M}{2}\ln 2\pi - \frac{1}{2}\ln|\bold K_{MM}| -\frac{1}{2} \bold u^T \bold K_{MM}^{-1} \bold u \right\} \\[10pt]
&= -\frac{1}{2}  \frac{\partial}{\partial \bold u} \left(  \bold u^T \bold K_{MM}^{-1} \bold u \right) \\[10pt]
&= - \bold K^{-1}_{MM} \bold u
\end{align*}
$$

となるので、

$$
\begin{align*}
\frac{\partial \ln p(\bold u | \bold y)}{\partial \bold u}
&= \frac{\partial}{\partial \bold u} \{\ln p(\bold y | \bold u)\} + \frac{\partial}{\partial \bold u}\{ \ln p(\bold u)\} \\[10pt]
&= \bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y - \bold K^{-1}_{MM} \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \bold K^{-1}_{MM} \bold u - \bold K^{-1}_{MM} \bold u \\[10pt]
&= - \{\bold K^{-1}_{MM} \bold K_{MM} \bold K^{-1}_{MM} + \bold K^{-1}_{MM} \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \bold K^{-1}_{MM} \}\bold u + \bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y \\[10pt]
&= - [\bold K^{-1}_{MM} \{\bold K_{MM} + \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \} \bold K^{-1}_{MM} ]\bold u + \bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y \\[10pt]
&= - (\bold K^{-1}_{MM} \bold Q_{MM} \bold K^{-1}_{MM} )\bold u + \bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y \\[10pt]
&= - \widehat{\bold \Sigma}_{\bold u}^{-1} \bold u + \bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y \\[10pt]
&= - \widehat{\bold \Sigma}_{\bold u}^{-1} \bold u + \widehat{\bold \Sigma}_{\bold u}^{-1}\widehat{\bold \Sigma}_{\bold u} \bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y \\[10pt]
&= - \widehat{\bold \Sigma}_{\bold u}^{-1} \bold u + \widehat{\bold \Sigma}_{\bold u}^{-1} \bold K_{MM} \bold Q^{-1}_{MM} \bold K_{MM} \bold K^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y \\[10pt]
&= - \widehat{\bold \Sigma}_{\bold u}^{-1} \bold u + \widehat{\bold \Sigma}_{\bold u}^{-1} \bold K_{MM} \bold Q^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y \\[10pt]
&= - \widehat{\bold \Sigma}_{\bold u}^{-1} \bold u + \widehat{\bold \Sigma}_{\bold u}^{-1} \widehat{\bold u}, \\[15pt]
\text{where}& \\
\bold Q_{MM} &= \bold K_{MM} + \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \\[10pt]
\widehat{\bold \Sigma}_{\bold u} &= \bold K_{MM} \bold Q^{-1}_{MM} \bold K_{MM} \\[10pt]
\widehat{\bold u} &= \bold K_{MM} \bold Q^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y
\end{align*}
$$

となります。一般の多変量ガウス分布の場合と見比べれば、

$$
\begin{align*}
p(\bold u| \bold y)
&= \mathcal N ( \widehat{\bold u}, \widehat{\bold \Sigma}_{\bold u})
\end{align*}
$$

となって補助変数の事後確率が得られました。

### 予測分布 $p(\bold f^*| \bold y)$

ここで $p(\bold f | \bold y)$ と $p(\bold f^* | \bold y)$ の近似分布に戻ります。

まず $p(\bold f | \bold y) \approx \int p(\bold f| \bold u) p(\bold u | \bold y) d\bold u$ について、

$$
\begin{align*}
\bold u | \bold y &\sim \mathcal N (\bold u | \widehat{\bold u}, \widehat{\bold \Sigma}_{\bold u}) \\[10pt]
\bold f | \bold u &\sim \mathcal N (\bold f | (\bold K^T_{MN} \bold K^{-1}_{MM}) \bold u + \bold 0_N, \bold \Lambda)
\end{align*}
$$

なので、多変量ガウス分布の線形変換より

$$
\begin{align*}
p(\bold f | \bold y) &\approx \mathcal N (\bold f | (\bold K^T_{MN} \bold K^{-1}_{MM}) \widehat{\bold u} + \bold 0_N, \bold \Lambda + (\bold K^T_{MN} \bold K^{-1}_{MM}) \widehat{\bold \Sigma}_{\bold u} (\bold K^T_{MN} \bold K^{-1}_{MM})^T) \\[10pt]
&= \mathcal N (\bold f | (\bold K^T_{MN} \bold K^{-1}_{MM}) \widehat{\bold u}, \bold \Lambda + (\bold K^T_{MN} \bold K^{-1}_{MM}) \widehat{\bold \Sigma}_{\bold u} (\bold K^T_{MN} \bold K^{-1}_{MM})^T) \\[10pt]
&= \mathcal N (\bold f | (\bold K^T_{MN} \bold K^{-1}_{MM}) \widehat{\bold u}, \bold \Lambda + (\bold K^T_{MN} \bold K^{-1}_{MM}) \bold K_{MM} \bold Q_{MM}^{-1} \bold K_{MM}^T (\bold K^T_{MN} \bold K^{-1}_{MM})^T) \\[10pt]
&= \mathcal N (\bold f | (\bold K^T_{MN} \bold K^{-1}_{MM}) \widehat{\bold u}, \bold \Lambda + \bold K^T_{MN} \bold Q_{MM}^{-1} \bold K_{MN}) \\[15pt]
\text{where}& \\
\bold \Lambda &= \text{ diag}(\bold K_{NN} - \bold K^T_{MN} \bold K^{-1}_{MM} \bold K_{MN}) \\[10pt]
\bold Q_{MM} &= \bold K_{MM} + \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \\[10pt]
\widehat{\bold u} &= \bold K_{MM} \bold Q^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y
\end{align*}
$$

となります。

$p(\bold f^* | \bold y) \approx \int p(\bold f^*| \bold u) p(\bold u | \bold y) d\bold u$ も同様に考えれば、

$$
\begin{align*}
p(\bold f^* | \bold y)
&\approx \mathcal N (\bold f^* | (\bold K^T_{M*} \bold K^{-1}_{MM}) \widehat{\bold u}, \bold \Lambda_* + \bold K^T_{M*} \bold Q_{MM}^{-1} \bold K_{M*}) \\[15pt]
\text{where}& \\
\bold \Lambda_* &= \text{ diag}(\bold K_{**} - \bold K^T_{M*} \bold K^{-1}_{MM} \bold K_{M*}) \\[10pt]
\bold Q_{MM} &= \bold K_{MM} + \bold K_{MN}(\bold \sigma^2 \bold I + \bold\Lambda)^{-1} \bold K^T_{MN} \\[10pt]
\widehat{\bold u} &= \bold K_{MM} \bold Q^{-1}_{MM} \bold K_{MN}(\sigma^2 \bold I + \bold \Lambda)^{-1} \bold y
\end{align*}
$$

と予測分布が求まります。

## 2.2 ハイパーパラメータの最適化

### 周辺尤度 $p(\bold y)$

ハイパーパラメータの最適化にそのまま使用するわけではありませんが、周辺尤度 $p(\bold y)$ 、

$$
p(\bold y)
= \int p(\bold y| \bold f) p(\bold f| \bold u) p(\bold u) d\bold f d \bold u
$$

つまり補助変数法の確率的生成モデルのエビデンスについても求めておきます。まずは 

$$
\begin{align*}
\bold u &\sim \mathcal N (\bold u| \bold 0_M, \bold K_{MM}) \\[10pt]
\bold f | \bold u &\sim \mathcal N (\bold f|\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda) = \mathcal N (\bold f| (\bold K^T_{MN} \bold K^{-1}_{MM}) \bold u + \bold 0_N, \bold \Lambda)
\end{align*}
$$

を見ると、 $\bold f| \bold u$ はアフィン変換 $(\bold K^T_{MN} \bold K^{-1}_{MM}) \bold u + \bold 0_N$ を平均とした精度 $\bold \Lambda$ の多変量ガウス分布に従っているので、多変量ガウス分布の線形変換により 

$$
\begin{align*}
p(\bold f)
&= \int p(\bold f| \bold u)p(\bold u) d\bold u \\[10pt]
&= \mathcal N (\bold f| (\bold K^T_{MN} \bold K^{-1}_{MM}) \bold 0_M + \bold 0_N, \bold \Lambda + (\bold K^T_{MN} \bold K^{-1}_{MM}) \bold K_{MM} (\bold K^T_{MN} \bold K^{-1}_{MM})^T ) \\[10pt]
&= \mathcal N (\bold f| (\bold 0_N, \bold \Lambda +  \bold K^T_{MN} \bold K^{-1}_{MM}\bold K_{MN})
\end{align*}
$$

が得られ、畳み込みにより

$$
\begin{align*}
p(\bold y)
&= \int p(\bold y | \bold f) p(\bold f) d \bold f  \\[10pt]
&= \int \mathcal N (\bold y |\bold f, \sigma^2 \bold I) \mathcal N (\bold f| (\bold 0_N, \bold \Lambda +  \bold K^T_{MN} \bold K^{-1}_{MM}\bold K_{MN}) d\bold f \\[10pt]
&= \int \mathcal N (\bold y - \bold f |\bold 0_N, \sigma^2 \bold I) \mathcal N (\bold f| (\bold 0_N, \bold \Lambda +  \bold K^T_{MN} \bold K^{-1}_{MM}\bold K_{MN}) d\bold f \\[10pt]
&= \mathcal N (\bold y| \bold 0_N  + \bold 0_N, \sigma^2\bold I + \bold \Lambda + \bold K^T_{MN} \bold K^{-1}_{MM}\bold K_{MN}) \\[10pt]
&= \mathcal N (\bold y| \bold 0_N , \sigma^2\bold I + \bold \Lambda + \bold K^T_{MN} \bold K^{-1}_{MM}\bold K_{MN})
\end{align*}
$$

となり、目的のエビデンスが得られます

### ELBO

最適化の場合にはこちらを目的関数として用います。先ほど上で考えた周辺尤度の対数をとれば

$$
\begin{align*}
\log p(\bold y| \bold X)
&= \log \int p(\bold y| \bold f) p(\bold f| \bold u) p(\bold u) d\bold f d \bold u 
\end{align*}
$$

となり、ここから導出したいのはこれを下から抑える表現です。まず、イェンセンの不等式を用いて

$$
\begin{align*}
\log p(\bold y| \bold u)
&= \log \int p(\bold y| \bold f) p(\bold f| \bold u) d\bold f \\[10pt]
&\geq \int \log\{p(\bold y | \bold f)\} p(\bold f| \bold u) d\bold f \eqqcolon \mathcal L_1\\[10pt]
&= \int \left\{-\frac{N}{2}\log 2\pi -\frac{1}{2}\log |\sigma^2 \bold I| - \frac{1}{2}(\bold y - \bold f)^T(\sigma^2 \bold I)^{-1}(\bold y - \bold f) \right\} \mathcal N (\bold f|\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda) d\bold f \\[10pt]
&= \int \left( -\frac{N}{2}\log 2\pi -\frac{1}{2}\log |\sigma^2 \bold I| \right)\mathcal N (\bold f|\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda)d\bold f + \int \left\{ - \frac{1}{2}(\bold y - \bold f)^T(\sigma^2 \bold I)^{-1}(\bold y - \bold f) \right\}\mathcal N (\bold f|\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda)d\bold f \\[10pt]
&= -\frac{N}{2}\log 2\pi -\frac{1}{2}\log |\sigma^2 \bold I|
-\frac{1}{2} \int (\bold f - \bold y)^T(\sigma^2 \bold I)^{-1}(\bold f - \bold y) \mathcal N (\bold f|\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda)d\bold f
\end{align*}
$$

ここで、 $\int (\bold f - \bold y)^T(\sigma^2 \bold I)^{-1}(\bold f - \bold y) \mathcal N (\bold f|\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda)d\bold f$ は二次形式の期待値で、

$$
\begin{align*}
\int (\bold f - \bold y)^T & (\sigma^2 \bold I)^{-1}(\bold f - \bold y) \mathcal N (\bold f|\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda)d\bold f \\[10pt]
&= \text{tr}((\sigma^2 \bold I)^{-1} \bold \Lambda) + (\bold K^T_{MN} \bold K^{-1}_{MM} \bold u - \bold y)^T (\sigma^2 \bold I)^{-1} (\bold K^T_{MN} \bold K^{-1}_{MM} \bold u - \bold y) \\[10pt]
&= \frac{1}{\sigma^2}\text{tr}(\bold \Lambda) + (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)^T (\sigma^2 \bold I)^{-1} (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)
\end{align*}
$$

となるので、

$$
\begin{align*}
\mathcal L_1
&= -\frac{N}{2}\log 2\pi -\frac{1}{2}\log |\sigma^2 \bold I| -\frac{1}{2} \int (\bold f - \bold y)^T(\sigma^2 \bold I)^{-1}(\bold f - \bold y) \mathcal N (\bold f|\bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \bold \Lambda)d\bold f \\[10pt]
&= -\frac{N}{2}\log 2\pi -\frac{1}{2}\log |\sigma^2 \bold I| - \frac{1}{2} \left\{ \frac{1}{\sigma^2}\text{tr}(\bold \Lambda) + (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)^T (\sigma^2 \bold I)^{-1} (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u) \right \} \\[10pt]
&= -\frac{N}{2}\log 2\pi -\frac{1}{2}\log |\sigma^2 \bold I| - \frac{1}{2\sigma^2}\text{tr}(\bold \Lambda) -\frac{1}{2} (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)^T (\sigma^2 \bold I)^{-1} (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u) \\[10pt]
&= -\frac{N}{2}\log 2\pi -\frac{1}{2}\log |\sigma^2 \bold I| -\frac{1}{2} (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u)^T (\sigma^2 \bold I)^{-1} (\bold y - \bold K^T_{MN} \bold K^{-1}_{MM} \bold u) - \frac{1}{2\sigma^2}\text{tr}(\bold \Lambda) \\[10pt]
&= \log \mathcal N (\bold y| \bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \sigma^2 \bold I)- \frac{1}{2\sigma^2}\text{tr}(\bold \Lambda) 
\end{align*}
$$

となります。最初の対数周辺尤度に戻って、得られた $p(\bold y| \bold u)$ の下界を代入すると対数周辺尤度の下界の表現が得られて、これを ELBO とします。

$$
\begin{align*}
\log p(\bold y| \bold X)
&= \log \int p(\bold y| \bold f) p(\bold f| \bold u) p(\bold u) d\bold f d \bold u \\[10pt]
&= \log \int p(\bold y| \bold u) p(\bold u) d \bold u \\[10pt]
&\geq \log \int \exp(\mathcal L_1) p(\bold u) d\bold u \eqqcolon \mathcal L_2
\end{align*}
$$

$$
\begin{align*}
\mathcal L_2
&= \log \int \exp(\mathcal L_1) p(\bold u) d\bold u  \\[10pt]
&= \log \int \mathcal N (\bold y| \bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \sigma^2 \bold I) \exp\left\{- \frac{1}{2\sigma^2}\text{tr}(\bold \Lambda) \right\}  \mathcal N (\bold u| \bold 0_M, \bold K_{MM}) d\bold u  \\[10pt]
&= \log \int \mathcal N (\bold y| \bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \sigma^2 \bold I) \mathcal N (\bold u| \bold 0_M, \bold K_{MM}) d\bold u - \frac{1}{2\sigma^2}\text{tr}(\bold \Lambda) 
\end{align*}
$$

ここで、周辺尤度を求めたときに使用したアフィン変換のロジックをもう一度使えば

$$
\begin{align*}
\int \mathcal N (\bold y| \bold K^T_{MN} & \bold K^{-1}_{MM} \bold u, \sigma^2 \bold I) \mathcal N (\bold u| \bold 0_M, \bold K_{MM}) d\bold u \\[10pt]
&= \int \mathcal N (\bold y| (\bold K^T_{MN} \bold K^{-1}_{MM}) \bold u + \bold 0_N, \sigma^2 \bold I)\mathcal N (\bold u| \bold 0_M, \bold K_{MM}) d\bold u \\[10pt]
&= \mathcal N (\bold y| (\bold K^T_{MN} \bold K^{-1}_{MM}) \bold 0_M + \bold 0_N, \sigma^2 \bold I + (\bold K^T_{MN} \bold K^{-1}_{MM})\bold K_{MM} (\bold K^T_{MN} \bold K^{-1}_{MM})^T) \\[10pt]
&= \mathcal N (\bold y| \bold 0_N, \bold K^T_{MN} \bold K^{-1}_{MM}\bold K_{MN} + \sigma^2 \bold I)
\end{align*}
$$

となるので、これを代入して、

$$
\begin{align*}
\mathcal L_2
&= \log \int \mathcal N (\bold y| \bold K^T_{MN} \bold K^{-1}_{MM} \bold u, \sigma^2 \bold I) \mathcal N (\bold u| \bold 0_M, \bold K_{MM}) d\bold u - \frac{1}{2\sigma^2}\text{tr}(\bold \Lambda)  \\[10pt]
&= \log \mathcal N (\bold y| \bold 0_N, \bold K^T_{MN} \bold K^{-1}_{MM}\bold K_{MN} + \sigma^2 \bold I)- \frac{1}{2\sigma^2}\text{tr}(\bold \Lambda)
\end{align*}
$$

となり、これで ELBO の導出は完了です。

# 3. 数学の補足

## 3.1 多変量正規分布の対数とその微分

$D$ 次元の多変量正規分布の確率密度関数は、

$$
\mathcal N (\bold x | \boldsymbol \mu, \bold \Sigma) 
= \frac{1}{\sqrt{(2\pi)^D |\bold \Sigma|}} \exp\left\{ -\frac{1}{2}(\bold x - \boldsymbol \mu)^T \bold \Sigma^{-1} (\bold x - \boldsymbol \mu) \right\}
$$

となので、対数をとると

$$
\begin{align*}
\ln \mathcal N (\bold x | \boldsymbol \mu, \bold \Sigma) 
&= \ln \left[\frac{1}{\sqrt{(2\pi)^D |\bold \Sigma|}} \exp\left\{ -\frac{1}{2}(\bold x - \boldsymbol \mu)^T \bold \Sigma^{-1} (\bold x - \boldsymbol \mu) \right\}\right] \\[15pt]
&= - \frac{D}{2}\ln{2\pi} - \frac{1}{2} \ln{|\bold \Sigma|} -\frac{1}{2}(\bold x - \boldsymbol \mu)^T \bold \Sigma^{-1} (\bold x - \boldsymbol \mu)
\end{align*}
$$

となる。これを $\bold x$ について微分すると

$$
\begin{align*}
\frac{\partial}{\partial \bold x} \{\ln \mathcal N (\bold x | \boldsymbol \mu, \bold \Sigma) \}
&=\frac{\partial}{\partial \bold x} \left \{- \frac{D}{2}\ln{2\pi} - \frac{1}{2} \ln{|\bold \Sigma|} -\frac{1}{2}(\bold x - \boldsymbol \mu)^T \bold \Sigma^{-1} (\bold x - \boldsymbol \mu) \right \} \\[10pt]
&=\frac{\partial}{\partial \bold x} \left \{-\frac{1}{2}(\bold x - \boldsymbol \mu)^T \bold \Sigma^{-1} (\bold x - \boldsymbol \mu) \right \} \\[10pt]
&=\frac{\partial}{\partial \bold x} \left \{-\frac{1}{2}(\bold x^T \bold \Sigma^{-1} \bold x - 2 \boldsymbol \mu^T \bold \Sigma^{-1} \bold x + \boldsymbol \mu^T \bold \Sigma^{-1} \boldsymbol \mu) \right \} \\[10pt]
&= -\frac{1}{2}\left\{(\bold \Sigma^{-1} + (\bold \Sigma^{-1})^T) \bold x - 2(\boldsymbol \mu^T \bold \Sigma^{-1})^T \right\} \\[10pt]
&= - \bold \Sigma^{-1} \bold x + \bold \Sigma^{-1} \boldsymbol \mu
\end{align*}
$$

となる。確率変数が多変量正規分布に従うことが分かっているとき、確率密度関数全体を計算する代わりに部分的に計算して一般の式と比較すると楽に望む結果が得られることがある。

## 3.2 多変量ガウス分布の畳み込み

### 確率分布の畳み込み

確率分布が連続の場合のみを紹介します。まず、２つの独立な確率分布 $X$ と $Y$ から、 $Z = X + Y$ という確率分布を考えます。ここで $Z$ の累積分布関数は、

$$
\begin{align*}
F_Z(z)
&= P(Z \leq z) \\[10pt]
&= P(X + Y \leq z) \\[10pt]
&= \int_{x \in \Omega_X} P(X + Y \leq z| X = x) f_X(x) dx \\[10pt]
&= \int_{x \in \Omega_X} P(Y \leq z - x| X = x) f_X(x) dx \\[10pt]
&= \int_{x \in \Omega_X} P(Y \leq z - x) f_X(x) dx \quad(\because X \perp \!\!\!\! \perp Y)\\[10pt]
&= \int_{x \in \Omega_X} F_Y(z - x) f_X(x) dx 
\end{align*}
$$

となるので、 $z$ について微分すれば

$$
\begin{align*}
f_Z(z)
&= \frac{d}{dz}F_Z(z) \\[10pt]
&= \int_{x \in \Omega_X} f_X(x) f_Y(z - x) dx 
\end{align*}
$$

となり、新たな確率変数 $Z$ の確率密度関数が得られました。

### 多変量ガウス分布の畳み込み

２つの独立な $D$ 次元の多変量正規分布 $\bold X$ と $\bold Y$ を考えます。その確率密度関数はそれぞれ

$$
\begin{align*}
\mathcal N (\bold x | \boldsymbol \mu_1, \bold \Sigma_1) 
&= \frac{1}{\sqrt{(2\pi)^D |\bold \Sigma_1|}} \exp\left\{ -\frac{1}{2}(\bold x - \boldsymbol \mu_1)^T \bold \Sigma_1^{-1} (\bold x - \boldsymbol \mu_1) \right\} \\[10pt]
\mathcal N (\bold y | \boldsymbol \mu_2, \bold \Sigma_2) 
&= \frac{1}{\sqrt{(2\pi)^D |\bold \Sigma_2|}} \exp\left\{ -\frac{1}{2}(\bold y - \boldsymbol \mu_2)^T \bold \Sigma_2^{-1} (\bold y - \boldsymbol \mu_2) \right\}
\end{align*}
$$

として、新たな確率変数 $\bold Z = \bold X + \bold Y$ の従う確率密度関数を導出します。

まずは、一般の式から変形していきます。

$$
\begin{align*}
f_{\bold Z}(\bold z)
&= \int_{\bold x \in \Omega_{\bold X}} f_{\bold X}(\bold x) f_{\bold Y}(\bold y) d\bold x \\[10pt]
&= \int_{\bold x \in \Omega_{\bold X}} f_{\bold X}(\bold x) f_{\bold Y}(\bold z - \bold x) d\bold x \\[10pt]
&= \int_{\bold x \in \Omega_{\bold X}} \mathcal N (\bold x | \boldsymbol \mu_1, \bold \Sigma_1) \mathcal N (\bold z - \bold x | \boldsymbol \mu_2, \bold \Sigma_2) d\bold x \\[10pt]
&= \int_{\bold x \in \Omega_{\bold X}} \frac{1}{\sqrt{(2\pi)^D |\bold \Sigma_1|}} \exp\left\{ -\frac{1}{2}(\bold x - \boldsymbol \mu_1)^T \bold \Sigma_1^{-1} (\bold x - \boldsymbol \mu_1) \right\}
\cdot \frac{1}{\sqrt{(2\pi)^D |\bold \Sigma_2|}} \exp\left\{ -\frac{1}{2}(\bold z - \bold x - \boldsymbol \mu_2)^T \bold \Sigma_2^{-1} (\bold z - \bold x - \boldsymbol \mu_2) \right\} 
 d\bold x \\[10pt]
&= \frac{1}{\sqrt{(2\pi)^D}} \int_{\bold x \in \Omega_{\bold X}} \frac{1}{\sqrt{(2\pi)^D |\bold \Sigma_1||\bold \Sigma_2|}} \exp\left\{ -\frac{1}{2}\left( (\bold x - \boldsymbol \mu_1)^T \bold \Sigma_1^{-1} (\bold x - \boldsymbol \mu_1) + (\bold z - \bold x - \boldsymbol \mu_2)^T \bold \Sigma_2^{-1} (\bold z - \bold x - \boldsymbol \mu_2) \right) \right\} 
 d\bold x 
\end{align*}
$$

ここで、 $\bold A$ と $\bold B$ が対称行列のとき、

$$
\begin{align*}
(\bold x - \bold a)^T & \bold A (\bold x - \bold a) + (\bold x - \bold b)^T \bold B (\bold x - \bold b) \\[10pt]
&= (\bold x - \bold c)^T (\bold A + \bold B)(\bold x - \bold c) + (\bold a - \bold b)^T \bold C (\bold a - \bold b) \\[15pt]
\text{where} &\quad \\
\bold c &= (\bold A + \bold B)^{-1} (\bold A \bold a + \bold B \bold b) \\[10pt]
\bold C &= \bold A(\bold A + \bold B)^{-1} \bold B = (\bold A^{-1} + \bold B^{-1})^{-1}
\end{align*}
$$

なので、

$$
\begin{align*}
(\bold x - \boldsymbol \mu_1)^T & \bold \Sigma_1^{-1} (\bold x - \boldsymbol \mu_1) + (\bold z - \bold x - \boldsymbol \mu_2)^T \bold \Sigma_2^{-1} (\bold z - \bold x - \boldsymbol \mu_2) \\[10pt]
&= \left\{\bold x - (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}(\bold \Sigma_1^{-1}\boldsymbol \mu_1 + \bold \Sigma_2^{-1} \boldsymbol \mu_2)\right\}^T \left\{ (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1} \right\}^{-1} \left\{ \bold x - (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}(\bold \Sigma_1^{-1}\boldsymbol \mu_1 + \bold \Sigma_2^{-1} \boldsymbol \mu_2)\right\} \\[10pt]
&\quad + \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\}^T (\bold \Sigma_1 + \bold \Sigma_2)^{-1} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\}
\end{align*}
$$

となり、

$$
\begin{align*}
f_{\bold Z}(\bold z)
&= \frac{1}{\sqrt{(2\pi)^D}} \int_{\bold x \in \Omega_{\bold X}} \frac{1}{\sqrt{(2\pi)^D |\bold \Sigma_1||\bold \Sigma_2|}} \exp\left\{ -\frac{1}{2}\left( (\bold x - \boldsymbol \mu_1)^T \bold \Sigma_1^{-1} (\bold x - \boldsymbol \mu_1) + (\bold z - \bold x - \boldsymbol \mu_2)^T \bold \Sigma_2^{-1} (\bold z - \bold x - \boldsymbol \mu_2) \right) \right\} 
 d\bold x \\[20pt]
&= \frac{\sqrt{|\bold \Sigma_1 + \bold \Sigma_2|}}{\sqrt{(2\pi)^D}\sqrt{|\bold \Sigma_1 + \bold \Sigma_2|}} \exp\left\{ -\frac{1}{2} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\}^T (\bold \Sigma_1 + \bold \Sigma_2)^{-1} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\} \right\} \\[15pt]
&\quad\quad \times \int_{\bold x \in \Omega_{\bold X}} \frac{1}{\sqrt{(2\pi)^D |\bold \Sigma_1||\bold \Sigma_2|}} \exp\left\{ -\frac{1}{2}\left\{\bold x - (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}(\bold \Sigma_1^{-1}\boldsymbol \mu_1 + \bold \Sigma_2^{-1} \boldsymbol \mu_2)\right\}^T \left\{ (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1} \right\}^{-1} \left\{ \bold x - (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}(\bold \Sigma_1^{-1}\boldsymbol \mu_1 + \bold \Sigma_2^{-1} \boldsymbol \mu_2)\right\} \right\} 
 d\bold x \\[20pt]
&= \frac{1}{\sqrt{(2\pi)^D|\bold \Sigma_1 + \bold \Sigma_2|}} \exp\left\{ -\frac{1}{2} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\}^T (\bold \Sigma_1 + \bold \Sigma_2)^{-1} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\} \right\} \\[15pt]
&\quad\quad \times \int_{\bold x \in \Omega_{\bold X}} \frac{1}{\sqrt{(2\pi)^D}} \sqrt{\frac{|\bold \Sigma_1 + \bold \Sigma_2|}{|\bold \Sigma_1||\bold \Sigma_2|}} \exp\left\{ -\frac{1}{2}\left\{\bold x - (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}(\bold \Sigma_1^{-1}\boldsymbol \mu_1 + \bold \Sigma_2^{-1} \boldsymbol \mu_2)\right\}^T \left\{ (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1} \right\}^{-1} \left\{ \bold x - (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}(\bold \Sigma_1^{-1}\boldsymbol \mu_1 + \bold \Sigma_2^{-1} \boldsymbol \mu_2)\right\} \right\} 
 d\bold x \\[20pt]
&= \frac{1}{\sqrt{(2\pi)^D|\bold \Sigma_1 + \bold \Sigma_2|}} \exp\left\{ -\frac{1}{2} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\}^T (\bold \Sigma_1 + \bold \Sigma_2)^{-1} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\} \right\} \\[15pt]
&\quad\quad \times \int_{\bold x \in \Omega_{\bold X}} \frac{1}{\sqrt{(2\pi)^D|(\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}|}} \exp\left\{ -\frac{1}{2}\left\{\bold x - (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}(\bold \Sigma_1^{-1}\boldsymbol \mu_1 + \bold \Sigma_2^{-1} \boldsymbol \mu_2)\right\}^T \left\{ (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1} \right\}^{-1} \left\{ \bold x - (\bold \Sigma_1^{-1} + \bold \Sigma_2^{-1})^{-1}(\bold \Sigma_1^{-1}\boldsymbol \mu_1 + \bold \Sigma_2^{-1} \boldsymbol \mu_2)\right\} \right\} 
 d\bold x \\[20pt]
&= \frac{1}{\sqrt{(2\pi)^D|\bold \Sigma_1 + \bold \Sigma_2|}} \exp\left\{ -\frac{1}{2} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\}^T (\bold \Sigma_1 + \bold \Sigma_2)^{-1} \left\{ \bold z - (\boldsymbol \mu_1 + \boldsymbol \mu_2)\right\} \right\} \\[20pt]
&= \mathcal N(\bold z | \boldsymbol \mu_1 + \boldsymbol \mu_2, \bold \Sigma_1 + \bold \Sigma_2)
\end{align*}
$$

となって、 $\bold Z$ の確率密度関数が得られた。

## 3.3 多変量ガウス分布の線形変換

平均 $\boldsymbol \mu$ 、共分散 $\bold \Sigma_x$ をもつガウス分布に従う確率変数 $\bold x$ に対して、アフィン変換 $\bold W \bold x + \bold b$ を平均とした精度 $\bold \Sigma_y$ のガウス分布に従う確率変数 $\bold y$ を考えます。すなわち、

$$
\begin{align*}
p(\bold x) &= \mathcal N(\bold x| \boldsymbol \mu, \bold \Sigma_x) \\[10pt]
p(\bold y| \bold x) &= \mathcal N(\bold y | \bold W \bold x + \bold b, \bold \Sigma_y)
\end{align*}
$$

とするとき、 $p(\bold y)$ は

$$
\begin{align*}
p(\bold y) &= \mathcal N(\bold y | \bold W \boldsymbol \mu + \bold b, \bold \Sigma_y + \bold W \bold \Sigma_x \bold W^T)
\end{align*}
$$

となります。これの証明については、参照した「ベイズ深層学習」に載っている以上のことが書けなかったので与えないことにします。他には下のサイトにも載っています。

**[Linear Transformation of Gaussian Random Variable](https://stats.stackexchange.com/questions/173163/linear-transformation-of-gaussian-random-variable)**

## 3.4 共分散行列が零行列の多変量ガウス分布

たまに、計算をしているとガウス分布の分散が $0$ になってしまう時があります。例えば、ガウス過程回帰モデルの観測ノイズをカーネル関数に含めた形で表すと、

$$
y_n = f(\bold{x}_n), \text{ where } f \sim \text{GP}(\bold 0, k(\bold x, \bold x') + \sigma^2 \delta(n, n'))
$$

で、この周辺尤度を考えると、

$$
\begin{align*}
p(\bold y | \bold X)
&= \int p(\bold y, \bold f | \bold X) d\bold f \\
&= \int p(\bold y| \bold f) p(\bold f | \bold X) d \bold f \\
&= \int \mathcal N(\bold y | \bold f, \bold 0) \mathcal N(\bold f | \bold 0, \bold K_{NN} + \sigma^2 \bold I) d\bold f 
\end{align*}
$$

となって、 $\mathcal N(\bold y | \bold f, \bold 0)$ のように共分散行列が $\bold 0$ のガウス分布が出てきます。ガウス分布の確率密度関数を考えると、共分散行列の逆行列を求める操作が含まれるのでこれはおかしいのですが、 $\bold y = \bold f$ のときに必ず $1$ を返すと形式的にみなすことで、ある種の正規分布として考えられます。すると

$$
\begin{align*}
p(\bold y | \bold X)
&= \int \mathcal N(\bold y | \bold f, \bold 0) \mathcal N(\bold f | \bold 0, \bold K_{NN} + \sigma^2 \bold I) d\bold f \\[10pt]
&= \mathcal N(\bold y| \bold 0 + \bold 0, \bold 0 + \bold K_{NN} + \sigma^2 \bold I) \\[10pt]
&= \mathcal N(\bold y| \bold 0, \bold K_{NN} + \sigma^2 \bold I) 
\end{align*}
$$

となって、ノイズを含めない形の $\bold y$ の周辺尤度と同じものが得られました。

このような確率変数を “degenerate” あるいは、 “deterministic” などと言ったりするようです。

## 3.5 二次形式の期待値

期待値 $\boldsymbol \mu$ 、共分散行列 $\bold \Sigma$ の確率変数 $\bold X$ （n x 1 の確率変数ベクトル）があるとして、二次形式 $(\bold X - \bold a)^T \bold A (\bold X - \bold a)$ の期待値を考えます。ここで、 $\bold A$ は対象行列なので、

$$
(\bold X - \bold a)^T \bold A (\bold X - \bold a) = \bold X^T \bold A \bold X -2 \bold a^T \bold A \bold X + \bold a^T \bold A \bold a
$$

となり、

$$
\begin{align*}
\mathbb E[(\bold X - \bold a)^T \bold A (\bold X - \bold a)]
&= \mathbb E [\bold X^T \bold A \bold X -2 \bold a^T \bold A \bold X + \bold a^T \bold A \bold a] \\[10pt]
&= \mathbb E [\bold X^T \bold A \bold X] -2 \bold a^T \bold A \mathbb E[\bold X] + \bold a^T \bold A \bold a\mathbb E[1] \quad (\because \text{期待値の線形性}) \\[10pt]
&= \mathbb E [\bold X^T \bold A \bold X] -2 \bold a^T \bold A \boldsymbol \mu + \bold a^T \bold A \bold a
\end{align*}
$$

で、 $\mathbb E[\bold X^T \bold A \bold X]$ は、

$$
\begin{align*}
\mathbb E[\bold X^T \bold A \bold X] 
&= \mathbb E[\text{tr}(\bold X^T \bold A \bold X)] \quad (\because \bold X^T \bold A \bold X \text{はスカラー}) \\[10pt]
&= \mathbb E[\text{tr}(\bold A \bold X \bold X^T)] \quad (\because \text{tr}(ABC) = \text{tr}(BCA))\\[10pt]
&= \text{tr}( \bold A \mathbb E[\bold X \bold X^T]) \quad (\because \mathbb E[\text{tr}(A)] = \text{tr}(\mathbb E[A])) \\[10pt]
&= \text{tr}(\bold A [\text{Cov}(\bold X, \bold X) + \mathbb E[\bold X]\mathbb E[\bold X]^T) \\[10pt]
&= \text{tr}(\bold A(\bold \Sigma + \boldsymbol \mu \boldsymbol \mu^T)) \\[10pt]
&= \text{tr}(\bold A \bold \Sigma) + \text{tr}(\bold A \boldsymbol \mu \boldsymbol \mu^T) \\[10pt]
&= \text{tr}(\bold A \bold \Sigma) + \text{tr}(\boldsymbol \mu^T \bold A \boldsymbol \mu) \\[10pt]
&= \text{tr}(\bold A \bold \Sigma) + \boldsymbol \mu^T \bold A \boldsymbol \mu
\end{align*}
$$

なので、最終的には

$$
\begin{align*}
\mathbb E[(\bold X - \bold a)^T \bold A (\bold X - \bold a)]
&= \mathbb E [\bold X^T \bold A \bold X] -2 \bold a^T \bold A \boldsymbol \mu + \bold a^T \bold A \bold a \\[10pt]
&= \text{tr}(\bold A \bold \Sigma) + \boldsymbol \mu^T \bold A \boldsymbol \mu -2 \bold a^T \bold A \boldsymbol \mu + \bold a^T \bold A \bold a \\[10pt]
&= \text{tr}(\bold A \bold \Sigma) + (\boldsymbol \mu - \bold a)^T\bold A(\boldsymbol \mu - \bold a)
\end{align*}
$$

となります。

# 4. まとめ

この記事では、ガウス過程回帰の計算コスト削減アルゴリズムである補助変数法について紹介しました。当然のことながら、この記事で紹介したことは補助変数法のすべてではありません。ELBO をグローバル変数を明示的な形で定義することによって、データをミニバッチに分けてのハイパーパラメータの最適化が可能になり、より巨大なデータに立ち向かうことができるようになるようです。他にも「ガウス過程と機械学習（講談社）」では KISS-GP なる手法も紹介されています。

ガウス過程回帰に限らず、機械学習のアルゴリズム全般は、広い範囲の数学を用いて記述されているため、初学者にとってはなかなか勉強するのが大変だと思います。この記事が少しでもそんな人の助けになれば光栄です。

# 参照

1. [ガウス過程と機械学習](https://www.amazon.co.jp/%E3%82%AC%E3%82%A6%E3%82%B9%E9%81%8E%E7%A8%8B%E3%81%A8%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%83%95%E3%82%A7%E3%83%83%E3%82%B7%E3%83%A7%E3%83%8A%E3%83%AB%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E6%8C%81%E6%A9%8B-%E5%A4%A7%E5%9C%B0/dp/4061529269)
2. [ベイズ深層学習](https://www.amazon.co.jp/%E3%83%99%E3%82%A4%E3%82%BA%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%83%95%E3%82%A7%E3%83%83%E3%82%B7%E3%83%A7%E3%83%8A%E3%83%AB%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E9%A0%88%E5%B1%B1-%E6%95%A6%E5%BF%97/dp/4065168708/ref=pd_lpo_1?pd_rd_w=W7DVc&content-id=amzn1.sym.d769922e-188a-40cc-a180-3315f856e8d6&pf_rd_p=d769922e-188a-40cc-a180-3315f856e8d6&pf_rd_r=JY2RKY1JNZFN2R6VQA3C&pd_rd_wg=O0MOT&pd_rd_r=794f9943-2d92-491e-bad7-82ffb079b10c&pd_rd_i=4065168708&psc=1)
3. [ベクトル・行列を含む微分](http://taustation.com/vector-matrix-differentiation/)
4. [Sparse Gaussian process](http://krasserm.github.io/2020/12/12/gaussian-processes-sparse/)
5. [Derivation of SGPR equation](https://gpflow.readthedocs.io/en/v1.5.1-docs/notebooks/theory/SGPR_notes.html)
6. [Sparse Gaussian Processes using Pseudo-inputs](http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf)
7. [Variational Learning of Inducing Variables in Sparse Gaussian Process](http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf)
8. [Variational Model Selection for Sparse Gaussian Process Regression](https://mtitsias.github.io/files/sparseGPv2.pdf)
9. [A Unifying View of Sparse Approximate Gaussian Process Regression](https://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf)
10. [Gaussian Processes for Big Data](https://arxiv.org/ftp/arxiv/papers/1309/1309.6835.pdf)
11. [Convolution integrals of Normal distribution
functions](http://web.ist.utl.pt/susanavinga/renyi/convolution_normal.pdf)
12. [Expectation of a quadratic form](https://statproofbook.github.io/P/mean-qf.html)
13. [Degenerate distribution](https://en.wikipedia.org/wiki/Degenerate_distribution)
14. [Linear Transformation of Gaussian Random Variable](https://stats.stackexchange.com/questions/173163/linear-transformation-of-gaussian-random-variable)