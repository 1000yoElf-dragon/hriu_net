# HRIU-net
<p>This work was basically inspired by two publications:</p>
<p>[1] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
"Swin Transformer: Hierarchical Vision Transformer using ShiftedWindows",
arXiv:2103.14030v2 [cs.CV] 17 Aug 2021. https://github.com/microsoft/Swin-Transformer.</p>
<p>[2] Duo Li, Jie Hu, Changhu Wang, Xiangtai Li, Qi She, Lei Zhu, Tong Zhang, Qifeng Chen
"Involution: Inverting the Inherence of Convolution for Visual Recognition",
arXiv:2103.06255v2 [cs.CV] 11 Apr 2021. https://github.com/d-li14/involution</p>
<p></p>
<p>The task is to improve image processing using attention-like neural networks.</p>
<p>In [1] self-attention technic was proposed to use inside non-overlapping windows. To provide information transfer between different windows they alternated layers with not shifted and shifted window's position.</p>
<p>In [2] authors used convolution with kernels dependent on features of target location.</p>
<p>I propose somehow to join this two approaches and another one which I called 'imprinting'.</p>
<p>Like in [1], for core blocks I use non-overlapping windows with alternated shifts. Output features obtained as linear combination of 'value' vectors like in self-attention approach.
'Value' vectors calculated as linear transformation of input features at single spatial point. But the scores for each vector in combination are derived differently.</p>
<p>In self-attention approach used in [1] contribution of point 'i' to point 'j' based on inner product of 'query' at 'j' and 'key' at 'i' and positional bias.
Both 'query' and 'key' depend on features in corresponding point: s_1 = q_1(f[j]) @ k_1(f[i]) + bias[i,j].</p>
<p>In [2] contribution to point 'j' depends on only features in 'j', not in 'i'. Not exactly, but the main idea corresponds the next expression: s_2 = q_2(f[j]) @ k_2[i,j] + bias[i,j],
where k2[i] - learnable vector dependent on relative position of points 'i' and 'j'.</p>
<p>Whereas in [2] scores depends on just features in target location why we mustn't use the opposite term with scores depndent on source location?
It is what I called 'imprinting', may be improperly: s_3 = q_3[i,j] @ k_3(f[i,j]) + bias[i,j].</p>
<p>Finally, let join all three terms: s = q_1(f[j]) @ k_1(f[i]) + q_2(f[j]) @ k_2[i,j] + q_3[i,j] @ k_3(f[i,j]) + bias[i,j]</p>
<p>One also can write: s = (q_1(f[j]) + c_1[i,j]) @ (k_1(f[i]) + c_2[i,j]) + bias[i,j], but this would lead to increased memory usage, so it had better use the previuos expression.</p>
<p>One could see that it is a hyperbolic formula with a lot of relative positional parameters, so I called this approach Hyperbolic Relative Imprinting Unit.</p>
<p>Note that if we keep just 'imprinting' terms we will be able to project input features not only to the same spatial shape but to an arbitrary shape,
because it isn't necessary to calculate query at target location. This can be useful for shrinking spatial dimensions.</p>
<p></p>
<p></p>
<p></p>
<p></p>
