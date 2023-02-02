.. _oscilacije_implementacija:

Имплементација
=================

Да бисмо реализовали предложени модел пригушених осцилација у једној димензији описан обичном диференцијалном једначином у секцији :ref:`oscilacije_uvod`, уместо библиотеке SCIANN користићемо нешто новију библиотеку DeepXDE :cite:t:`lu2021deepxde`. Идеја је да овим материјалом покријемо све значајне софтверске оквире који подржавају ФПНМ методологију у тренутку писања (2022-2023). Како аутори кажу, DeepXDE библиотека је дизајнирана да служи и као образовно средство које ће се користити у високом школству и као истраживачки алат за решавање проблема у компјутерским наукама и инжењерству. Конкретно, DeepXDE може да решава проблеме за унапред дате почетне и граничне услове, као и инверзне проблеме уз нека додатна мерења. Подржава домене сложене геометрије и омогућава да кориснички код буде компактан, веома сличан математичкој формулацији. 

У односу на SCIANN, приступ DeepXDE подразумева нешто виши степен апстракције, па тиме и лакоће употребе. На пример, компликације око постављања колокационих тачака у којима важе гранични и почетни услови, као и величина *batch*-a, не потпадају под бригу корисника, већ се сама библиотека стара о томе да специфицирани број тачака подлеже граничним условима. Тај ниво апстракције у неким специфичним случајевима може представљати препреку, али у великој већини случајева доприноси јаснијем дефинисању проблема.

На следећем листингу дати су значајни делови имплмементације:

.. code-block:: python
    :caption: Решење проблема пригушених осцилација у 1Д коришћењем DeepXDE библиотеке
    :linenos:

    import deepxde as dde
    import numpy as np
    import matplotlib.pyplot as plt

    m = 1
    mu = 0.1
    k = 2

    # Pocetni uslovi
    x0, v0 = 0, 2

    delta = mu / (2*m)
    w0 = np.sqrt(k/m)

    # Da li je tacka blizu t=0 (provera pocetnog uslova)
    def boundary_l(t, on_initial):
        return on_initial and np.isclose(t[0], 0)

    # Jednacina ODE
    def Ode(t,x):
            dxdt = dde.grad.jacobian(x,t)
            dxdtt = dde.grad.hessian(x,t)
            return m * dxdtt + mu * dxdt + k*x
        
    # x(0)=x0
    def bc_func1(inputs, outputs, X):
        return outputs - x0

    # x'(0)=v0
    def bc_func2(inputs, outputs, X):
        return dde.grad.jacobian(outputs, inputs, i=0,j=None) - v0

    # Resava se na domenu t=(0,10)
    interval = dde.geometry.TimeDomain(0, 10)

    # Pocetni uslovi
    ic1 = dde.icbc.OperatorBC(interval, bc_func1, boundary_l)
    ic2 = dde.icbc.OperatorBC(interval, bc_func2, boundary_l)

    # Definsanje problema, granicnih uslova, broja kolokacionih tacaka
    data = dde.data.TimePDE(interval, Ode, [ic1, ic2], 100, 20, solution=func, num_test=100)
        
    layers = [1] + [30] * 2 + [1]
    activation = "tanh"
    init = "Glorot uniform"
    net = dde.nn.FNN(layers, activation, init)

    model = dde.Model(data, net)

    model.compile("adam", lr=.001, loss_weights=[0.01, 1, 1], metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=30000)

    T = np.linspace(0, 10, 100).reshape(100,1)
    x_pred = model.predict(T)

На почетку се дефинишу импорти и константе проблема, као и додатне константе ``delta`` и ``w0`` које смо користили приликом извођења аналитичког решења. Функцију

.. code-block:: python

    def boundary_l(t, on_initial):
        return on_initial and np.isclose(t[0], 0)

ћемо искористити за тестирање да ли је дата колокациона тачка близу тачке ``t=0``, тј. да ли за дату тачку важи почетни услов. Употребићемо је при формирању почетних услова за позицију и брзину. Даље, као што назив сугерише, наредна функција

.. code-block:: python

    def Ode(t,x):
            dxdt = dde.grad.jacobian(x,t)
            dxdtt = dde.grad.hessian(x,t)
            return m * dxdtt + mu * dxdt + k*x

представља поставку проблема у свом извворном облику једначине :math:numref:`eq:oscilacije-dif`. Овде је очигледна једна од главних предности ФПНМ, тј. да неке додатне трансформације у интеграционим тачкама нису потребне, већ само поставка диференцијалне једначине у виду функције губитка и граничних услова у истом облику. Услужне методе ``dde.grad.jacobian`` и ``dde.grad.hessian`` враћају прве, односно друге изводе по улазним варијаблама примењуући тзв. аутоматску диференцијацију. Подразумевано се у позадини користи *Tensorflow* за тензорске операције ниског нивоа. 

Поставка два почетна услова у форми функције губитка дата је у следеће две методе, за координату и брзину респективно:

.. code-block:: python

    def bc_func1(inputs, outputs, X):
        return outputs - x0

    def bc_func2(inputs, outputs, X):
        return dde.grad.jacobian(outputs, inputs, i=0,j=None) - v0

Након поставке једнодимензионог временског домена у коме се проблем решава:

.. code-block:: python

    interval = dde.geometry.TimeDomain(0, 10)

можемо да формирамо и објекте граничних услова комбинујући функције губитка са функцијом локације ``boundary_l()``:

.. code-block:: python

    ic1 = dde.icbc.OperatorBC(interval, bc_func1, boundary_l)
    ic2 = dde.icbc.OperatorBC(interval, bc_func2, boundary_l)

Сада имамо све елементе да формирамо објекат проблема који решавамо. Овде ћемо то учинити методом ``dde.data.TimePDE`` за временски зависне проблеме:

.. code-block:: python
    
    data = dde.data.TimePDE(interval, Ode, [ic1, ic2], 100, 20, solution=func, num_test=100)

Специфицирамо редом рачунски домен, основну једначину, листу граничних услова, број колокационих тачака за основни домен (100), број колокационих тачака за граничне услове (20), егзактно решење (ако постоји) и број тестних тачака (за поређење са егзактним решењем). У овом примеру ћемо игнорисати егзактно решење. Одмах се види разлика у поставци у односу на SCIANN приступ у начину навођења колокационих тачака. Наиме, код DeepXDE колокационе тачке се не генеришу мануелно, већ се препушта библиотеци да то уради за нас, што указује на један виши ниво апстракције. 

Наредне линије кода конструишу неуронску мрежу која ће се користити као апроксимација проблема, са свим својим хипер-параметрима:

.. code-block:: python

    layers = [1] + [30] * 2 + [1]
    activation = "tanh"
    init = "Glorot uniform"
    net = dde.nn.FNN(layers, activation, init)

Наша мрежа има један улаз, један излаз и два скривена слоја од по 30 неурона, са активацијом скривених слојева у виду ``tanh`` функције и одговарајућом иницијализацијом. На крају, можемо да кренемо у обучавање, када спојивши проблем и генерисану неуронску мрежу формирамо модел:

.. code-block:: python

    model = dde.Model(data, net)
    model.compile("adam", lr=.001, loss_weights=[0.01, 1, 1], metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=30000)

Ово су стандардне методе које се широко користе у области дубоког учења, па је довољно само поменути да се наводи алгоритам оптимизације, брзина учења и начин прорачуна грешке која управља овим процесом. Специфичност за ФПНМ је што овде листом ``loss_weights`` можемо и да "пондеришемо" тежине основне диференцијалне једначине, првог и другог граничног услова, респективно. У наредној секцији :ref:`oscilacije_rezultati` ћемо размотрити решења за сва три случаја пригушеног осциловања. 
