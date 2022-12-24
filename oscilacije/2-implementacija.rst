.. _oscilacije_implementacija:

Имплементација
=================

Да бисмо реализовали предложени модел пригушених осцилација у једној димензији описан обичном диференцијалном једначином у секцији :ref:`oscilacije_uvod`, уместо библиотеке SCIANN користићемо нешто новију библиотеку DeepXDE :cite:t:`lu2021deepxde`. Идеја је да овим материјалом покријемо све значајне софтверске оквире који подржавају ФЗНН методологију у тренутку писања (2022-2023). Како аутори кажу, DeepXDE библиотека је дизајнирана да служи и као образовно средство које ће се користити у високом чколству, као и као истраживачки алат за решавање проблема у компјутерским наукама и инжењерству. Конкретно, DeepXDE може да решава проблеме за унапред дате почетне и граничне услове, као и инверзне проблеме уз нека додатна мерења. Подржава домене сложене геометрије и омогућава да кориснички код буде компактан, веома сличан математичкој формулацији. 

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
    w0 = np.sqrt(m / k)

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

На почетку се дефинишу импорти и константе проблема. 