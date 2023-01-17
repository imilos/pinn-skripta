.. _akustika_socivo:

Дводимензиони проблем са сочивом
======================================

Овај одељак се бави нешто модификованим `примером из туторијала <https://www.dealii.org/current/doxygen/deal.II/step_29.html>`_ за познати софтверски оквир за рад са методом коначних елемената **Deal.II**. Првобитна сврха овог примера је да симулира својства фокусирања ултразвучног таласа који генерише сочиво претварача са променљивом геометријом. Недавне примене у медицинском имиџингу користе ултразвучне таласе не само за сврхе снимања, већ и за изазивање одређених локалних ефеката у материјалу, као што су промене у оптичким својствима, које се затим могу мерити другим техникама снимања. Витални састојак ових метода је способност фокусирања интензитета ултразвучног таласа у одређеном делу материјала, идеално у тачки, како би се могла испитати својства материјала на тој локацији.

Међутим, карактеристична таласна дужина ултразвука нешто мања од феномена које смо до сада моделовали ФЗНМ методом, те би нам био потребан изузетно велики број колокационих тачака што продужава тренирање. Зато смо таласну дужину нешто повећали (смањили фреквенцију), док остале параметре нисмо мењали. 

Како бисмо извели једначине за овај проблем, звук узимамо као талас којим се шири промена притиска: 

.. math::
    \frac{\partial^2 U}{\partial t^2} - c^2 \Delta U = 0


где је *c* брзина звука (која се због једноставности узима константом), :math:`U = U(x,t),\;x \in \Omega,\;t\in\mathrm{R}`. Граница :math:`\Gamma=\partial\Omega` подељена је на два дела, и то :math:`\Gamma_1` и :math:`\Gamma_2=\Gamma\setminus\Gamma_1`, где :math:`\Gamma_1` представља сочиво, а :math:`\Gamma_2` апсорбујућу границу. Заправо, желимо да направимо такав гранични услов на :math:`\Gamma_2` тако да се опонаша знатно већи домен. На :math:`\Gamma_1`, претварач генерише таласе константне фреквенције :math:`\omega \gt 0` и константне јединичне амплитуде:

.. math::
    U(x,t) = \cos{\omega t}, \qquad x\in \Gamma_1

Пошто нема других (интерних или граничних) извора и пошто само извор емитује таласе фреквенције :math:`\omega`, дозвољено је да извршимо раздвајање променљивих :math:`U(x,t) = \textrm{Re}\left(u(x)\,e^{i\omega t})\right)`. Комплексна функција :math:`u(x)` описује просторну зависност амплитуде и фазе (релативно у односу на извор) таласа фреквенције :math:`\omega`, док је амлитуда вечличина која нас интересује. Ако овако формулисану функцију уврстимо у таласну једначину, видимо да за *u* имамо

.. math::
    -\omega^2 u(x) - c^2\Delta u(x) = 0, \qquad x \in \Omega, \\
    u(x) = 1,  \qquad x \in \Gamma_1.

For finding suitable conditions on \(\Gamma_2\) that model an absorbing boundary, consider a wave of the form \(V(x,t)=e^{i(k\cdot x -\omega t)}\) with frequency \({\omega}\) traveling in direction \(k\in {\mathrm{R}^2}\). In order for \(V\) to solve the wave equation, \(|k|={\frac{\omega}{c}}\) must hold. Suppose that this wave hits the boundary in \(x_0\in\Gamma_2\) at a right angle, i.e. \(n=\frac{k}{|k|}\) with \(n\) denoting the outer unit normal of \(\Omega\) in \(x_0\). Then at \(x_0\), this wave satisfies the equation  

\[
c (n\cdot\nabla V) + \frac{\partial V}{\partial t} = (i\, c\, |k| - i\, \omega) V = 0.
\]

Hence, by enforcing the boundary condition

\[
c (n\cdot\nabla U) + \frac{\partial U}{\partial t} = 0, \qquad x\in\Gamma_2,
\]


waves that hit the boundary \(\Gamma_2\) at a right angle will be perfectly absorbed. On the other hand, those parts of the wave field that do not hit a boundary at a right angle do not satisfy this condition and enforcing it as a boundary condition will yield partial reflections, i.e. only parts of the wave will pass through the boundary as if it wasn't here whereas the remaining fraction of the wave will be reflected back into the domain.
<p >If we are willing to accept this as a sufficient approximation to an absorbing boundary we finally arrive at the following problem for \(u\): 

\begin{eqnarray*}
-\omega^2 u - c^2\Delta u &amp;=&amp; 0, \qquad x\in\Omega,\\
c (n\cdot\nabla u) + i\,\omega\,u &amp;=&amp;0, \qquad x\in\Gamma_2,\\
u &amp;=&amp; 1,  \qquad x\in\Gamma_1.
\end{eqnarray*}

This is a Helmholtz equation (similar to the one in <a class="el" href="step_7.html">step-7</a>, but this time with ''the bad sign'') with Dirichlet data on \(\Gamma_1\) and mixed boundary conditions on \(\Gamma_2\). Because of the condition on \(\Gamma_2\), we cannot just treat the equations for real and imaginary parts of \(u\) separately. What we can do however is to view the PDE for \(u\) as a system of two PDEs for the real and imaginary parts of \(u\), with the boundary condition on \(\Gamma_2\) representing the coupling terms between the two components of the system. This works along the following lines: Let \(v=\textrm{Re}\;u,\; w=\textrm{Im}\;u\), then in terms of \(v\) and \(w\) we have the following system:  

\begin{eqnarray*}
  \left.\begin{array}{ccc}
    -\omega^2 v - c^2\Delta v &amp;=&amp; 0 \quad\\
    -\omega^2 w - c^2\Delta w &amp;=&amp; 0 \quad
  \end{array}\right\} &amp;\;&amp; x\in\Omega,
        \\
  \left.\begin{array}{ccc}
    c (n\cdot\nabla v) - \omega\,w &amp;=&amp; 0 \quad\\
    c (n\cdot\nabla w) + \omega\,v &amp;=&amp; 0 \quad
  \end{array}\right\} &amp;\;&amp; x\in\Gamma_2,
        \\
        \left.\begin{array}{ccc}
    v &amp;=&amp; 1 \quad\\
    w &amp;=&amp; 0 \quad
  \end{array}\right\} &amp;\;&amp; x\in\Gamma_1.
\end{eqnarray*}

