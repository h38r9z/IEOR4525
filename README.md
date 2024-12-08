java c
IEOR4525   Fall   2021   Final 
December   17,   2021
1 Feed-forward Neural Networks (22 pts) 
1.1          Architecture and Computation (10 pts) 
Consider   a   network   with   a   single   hidden   layer.   Following   information   is   given   to   you:
•    data:   x   ∈ Rdi          and   y   ∈ Rdo
• hidden   vector:   h(1) ∈ Rn
• output   vector:   ˆ(y) ∈ Rdo
•      activation   function:   σ   (applied to the hidden   layer   (i.e.    from input to   hidden   vector),   but   NOT   to   the   output   layer   (i.e.   from   hidden   vector   to   output)   of the   network)
Let   W   (1)   ,   b(1)    denote   the   weight   and   bias   in   the   hidden   layer,   and   W   (2)   ,   b(2)    denote   the   weight   and   bias in   the   output   layer.
1.    (4   pts)   Specify   the   shape   of   the   network’s   parameters   W   (1)   ,   b(1)   ,   W   (2)   ,   b(2)   .
2.    (2   pts)   Compute   the   total   number   of trainable   parameters   in   this   network.
3.    (2   pts)   Specify   the   forward   pass   computation,   including   how   to   compute   the   hidden   vector   h(1)      and output   vector   ˆ(y)   of the   network.
4.    (2   pts)   Compute   the   computational   complexity      (using   big-O   notation)   of   the   forward   pass,   for   the single   data   point   (x,y).
1.2 Activation Functions (12 pts) 
1.    (4   pts)   Write   out   the   function   expressions   for   the   sigmoid   and   ReLU   activations,   respectively.
2.    (6   pts)   ”Leaky   ReLU”   is   another   kind   of activaition   that   is   similar   to   ReLU,   with   the   expression
for   some   small   value   α   > 0.    Write   out   the   derivatives   of   sigmoid,   ReLU,   and   leaky   ReLU,   respectively.
3.    (2    pts)   By   comparing   the   derivatives   of   ReLU   and   leaky   ReLU,   what   do   you   think   could   be   the potential   advantage   of using   leaky   ReLU,   rather than   ReLU,   in   a   neural   network   model.    Explain why   you   think   so.      (Hint:    think   about   how   the   derivative   of   the   activation   function   influences   the   whole   backpropagation   process.)
2 PCA (16 pts) 
Consider   the   following   3   data   points   in   R2   :   x1    =   (−1, −1),   x2    =   (0,   0),   x3    =   (1, 1).
1.    (4   pts)   Show   the   first   principal   component   of   the   dataset.
2.    (6    pts)   Now   consider   projecting   each   data   point   onto   the   subspace   spanned   by   the   first   principal component.    What   are   the   coordinates   of   each   data   point   in   this   space?    And   what   is   the   variance   of the   projected   data?
3.    (6   pts)   Now   let   us   see   how   well   1-d   PCA   captures   the   original   data.    For   each   data   point,   compute   the square   difference   between   its   representation   in   terms   of the   first   principal   component   and   its   original   form.
3 Matrix Completion (16 pts) 
Let   Ω   be   the   subset   of   observed   indices   for   a   matrix   completion   problem.       Recall the following   matrix completion   optimization   problems:
Suppose we   are given the following   partially-observed   matrix  Suppose we   solve  
with   r   =   1.   Is   there   a   unique   solution?   If   yes,   what   is   it?   If   not,   explain   why.
2. (8 pts) Suppose we are given the same matrix as in the previous question, and we solve (1) with r = 2.
Is   there   a   unique   solution?   If   yes,   what   is   it?   If   not,   explain   why.
4 Clustering (16 pts) 
1.    (2   pts)   Write   a   high-level   description   of   the   repeated   steps   performed   by   the   k-m代 写IEOR4525 Fall 2021 FinalPython
代做程序编程语言eans   algorithm.
2.    (8 pts) Let’s consider   the   data   points   1,   2,   9, 12,   20   ∈ R   and   the   Euclidean   distance.    Apply the   k-means   algorithm   with   the   following   values   for   k   and   the   following   initializations.   Write   down   all   the   steps.(a)      μ   1    =   1,   μ2    = 20,   k   =   2   (b)      μ   1    = 9,   μ2    =   12,   k   =   2   (c)      μ   1      =   1,   μ2      = 2,   k   =   2
3.    (6   pts)   Hierarchical   clustering:
With the same data points as before, write down the   iterations   for   the   hierarchical   clustering   algorithm   in   the   following   cases:
(a)      Agglomerative   hierarchical   clustering   with   single   linkage
(b)      Agglomerative   hierarchical   clustering   with   complete   linkage
5 Tree-based Methods and Boosting (30 pts) 
5.1 Decision Tree (5 pts) Figure   1   shows   a   dataset   with   4   data   points.   Each   data   point   in   the   dataset   has   two   inputs   features   x   and y,   and   a   positive   (+)   or   negative   (−)   label,   as   depicted   in   the   figure.    Draw   a   decision   tree   which   correctly classifies   each   data   point   in   the   dataset.    (Note:    you   need   to   draw   the   diagram   of   the   decision   tree   in   your answer.   Please   do   NOT   simply   draw   the   decision   boundary   in   Figure   1.)

Figure   1:   Example   data   set
5.2 Universality of Decision Trees (10 pts) 
1.    Show   that   any   binary   classifier   g   :   {0, 1}D      →   {0, 1}   can   be   implemented   as   a   decision   tree   classifier.   That   is,   for   any   classifier   g   there   exists   a   decision   tree   classifier   T   with   k   nodes   n1   , . . . ,   nk       (each   ni   with   a   corresponding   threshold   ti      ),   such   that   g(x) = T(x)   for   all   x   ∈ {0, 1}D   .
2.   What   is   the   best   possible   bound   one   can   give   on   the   maximum   height   of   such   a   decision   tree   T   (from part   one)?   Give   an   example   of   g   that   achieves   this   bound.
5.3 Boosting (15 pts) 

Figure   2:   Sample   training   data   for   boosting   algorithmWe   here   study   how   boosting   algorithm   behaves   on   a   very   simple   classification   dataset   shown   in   Figure   2.   We   use   decision   stump   for   each   weak   classifier   hi.    Decision   stump   classifier   chooses   a   constant   value   c   and   classifies   all   points   where   x   > c   as   one   class   and   other   points   where   x   ≤ c   as   the   other   class.
1.    (3   pts)   What   is   the   initial   weight   for   each   data   point?
2.    (3 pts)   Suppose   that   the   decision   stump   is   trained   to   minimize   the   classification   error.   Write   down   the decision   boundary   for   the   first   decision   stump.   Indicate   the   positive   and   negative   side   of the   decision   boundary.    (There   might   be   multiple   valid   decision   stumps   for   this   problem.    You   only   need   to   write   one,   and   use   it   for   the   following   questions.)
3.    (3   pts)   Write   down   the   point   whose   weight   increases   during   the   first   iteration   of   the   boosting   process.
4.    (3   pts)   Write   down   the   weight   that   is   assigned   to   each   data   point   at   the   end   of the   first   iteration   of   boosting   algorithm.
5.    (3   pts)   Can   boosting   algorithm   perfectly   classify   all   the   data   points   shown   in   Figure   2?    If   no,   briefly explain   why.   If yes,   what   is   the   minimum   number   of iterations?






         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
