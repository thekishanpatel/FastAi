import numpy as np
from tkinter import *
import torch
import matplotlib; matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.rcParams.update({'font.size': 3.5})
font1 = ('Helvetica', 13, 'bold')
font2 = ('Helvetica', 10, 'bold')

gui = Tk()
gui.geometry('742x600+200+50')
gui.title('Polynomial Regression')

def mdl():
    global c, n, x1, x, y, d
    degree = deg.get(); d = int(degree)
    c1 = co.get(); c = [float(i) for i in (c1.split(sep = ' '))]; 
    c2 = c1.split(sep = " ");
    n1 = nnum.get(); n = int(n1);
    
    x1 = torch.ones(n, 2);
    x1[:, 0].uniform_(-1, 1);
    x1, indicies = torch.sort(x1, 0)
    
    c = torch.as_tensor(c);
    x = torch.ones(n, d + 1);
    for i in range(0, d + 1, 1):
        x[:, len(x[1]) - 1 - i] = pow(x1[:,0], i)
    y = x@c + torch.rand(n);
    
    s = 'x'
    p = ''
    for i in range(0, d + 1, 1):
        if (i < d):
            s1 = str(c2[i]) + s + '^' + str(d-i) + ' + '
        else:
            s1 = str(c2[i])
        p += s1
   
    parammlabel['text'] = p
    
    f = Figure(figsize = (1.6, 1.6), dpi = 200)
    f1 = f.add_subplot(111)
    f1.scatter(x1[:,0], y, s = 1.5)
    t = "Degree = " + str(d) + " Polynomial Model"
    f1.set_title(t)
    canv = FigureCanvasTkAgg(f, model)
    canv.get_tk_widget().grid(column = 0, row = 7, padx = 20, pady = 10, columnspan = 2, sticky = 'w')

def mse(y, yhat): return((y - yhat)**2).mean()

def quitc(): gui.destroy()

def update():
    yhat = x@a
    loss = mse(y, yhat)
    loss.backward() # The derivate
    with torch.no_grad():
      a.sub_(lr * a.grad)
      a.grad.zero_()
    return loss

def poly():
    global a, b, t, losso, lr
    b = ci.get(); b = [float(i) for i in (b.split(sep = " "))];
    a = torch.as_tensor(b); a = torch.nn.Parameter(a)
    lr = lrate.get(); lr = float(lr);
    stop = sc.get(); stop = float(stop);
    losso = 0; t = 0;
    
    while(True):
        l = update()
        if(abs(l - losso) <= stop): break
        losso = l
        t += 1
    
    b1 = np.array(a.data)
    b1 = [round(i, 2) for i in b1]
    s = 'x'
    p = ''
    for i in range(0, d + 1, 1):
        if (i < d):
            s1 = str(b1[i]) + s + '^' + str(d-i) + ' + '
        else:
            s1 = str(b1[i])
        p += s1
    
    rparammlabel['text'] = p
    
    ff = Figure(figsize = (1.6, 1.6), dpi = 200)
    ff1 = ff.add_subplot(111)
   
    ff1.scatter(x1[:,0], y, s = 1.5, c = "red");
    line, = ff1.plot(x1[:,0], x@a.data, linewidth = 1)
    ff1.set_title("Polynomial Regression")
    
    canv = FigureCanvasTkAgg(ff, regressor)
    canv.get_tk_widget().grid(column = 0, row = 7, padx = 20, pady = 10, columnspan = 2, sticky = 'w')


model = Frame(gui, bg = 'green', width = 400); model.pack(side=LEFT, fill = BOTH)
title1 = Label(model, text = "Let's Build a Model to Perform Regression on", font = font1, bg = 'green')
title1.grid(column = 0, row = 0, columnspan = 4)

deglabel = Label(model, text = "Enter Degree:-", font = font2, bg = 'green'); deglabel.grid(column = 0, row = 2, padx = 10, pady = 10, sticky = 'w')
deg = Entry(model, width = 15); deg.grid(column = 1, row = 2, padx = 10, pady = 10, sticky = 'w')

coelabel = Label(model, text = "Enter 'Deg + 1' coefficients,\nseparated by a space:-", font = font2, bg = 'green'); coelabel.grid(column = 0, row = 3, padx = 10, sticky = 'w')
co = Entry(model, width = 15); co.grid(column = 1, row = 3, padx = 10, sticky = 'w')

nlabel = Label(model, text = "Enter the size of Dataset:-", font = font2, bg = 'green'); nlabel.grid(column = 0, row = 4, padx = 10, pady = 10, sticky = 'w')
nnum = Entry(model, width = 15); nnum.grid(column = 1, row = 4, padx = 10, pady = 10, sticky = 'w')
                             
modelbut = Button(model, text = "Model", font = font2, bd = 2, relief = "raised", width = 20, comman = mdl); modelbut.grid(column = 0, row = 6, columnspan = 2, padx = 5)

paramframe = LabelFrame(model, bg = 'green', height = 20); paramframe.grid(column = 0, row = 8, columnspan = 2, padx = 10, pady = 10, sticky = 'nesw')
parammlabel = Label(paramframe, bg = 'green', font = font2); parammlabel.grid(column = 0, row = 0, sticky = 'e')

regressor = Frame(gui, bg = 'gray', width = 400); regressor.pack(side=RIGHT, fill = BOTH)
title2 = Label(regressor, text = "Let's get Parameters via Polynomial Regression", font = font1, bg = 'gray')
title2.grid(column = 0, row = 0, columnspan = 4)

cilabel = Label(regressor, text = "Enter Initial Coefficients:-", font = font2, bg = 'gray'); cilabel.grid(column = 0, row = 2, padx = 10, pady = 10, sticky = 'w')
ci = Entry(regressor, width = 15); ci.grid(column = 1, row = 2, padx = 10, pady = 10, sticky = 'w')

lrlabel = Label(regressor, text = "Enter Desired Learning Rate:-", font = font2, bg = 'gray'); lrlabel.grid(column = 0, row = 3, padx = 10, sticky = 'w')
lrate = Entry(regressor, width = 15); lrate.grid(column = 1, row = 3, padx = 10, sticky = 'w')

sclabel = Label(regressor, text = "Enter the Stop Condition:-", font = font2, bg = 'gray'); sclabel.grid(column = 0, row = 4, padx = 10, pady = 10, sticky = 'w')
sc = Entry(regressor, width = 15); sc.grid(column = 1, row = 4, padx = 10, pady = 10, sticky = 'w')

polybut = Button(regressor, text = "Regress", font = font2, bd = 2, relief = "raised", width = 20, comman = poly); polybut.grid(column = 0, row = 6, columnspan = 2, pady = 10, padx = 5)

rparamframe = LabelFrame(regressor, bg = 'gray', height = 20); rparamframe.grid(column = 0, row = 8, columnspan = 2, padx = 10, pady = 10, sticky = 'nesw')
rparammlabel = Label(rparamframe, bg = 'gray', font = font2); rparammlabel.grid(column = 0, row = 0, sticky = 'e')

qbut = Button(regressor, bg = "red", bd = 2, relief = "raised", text = "Exit Regressor", command = quitc); qbut.grid(column = 0, row = 10, columnspan = 2, padx = 5)

gui.mainloop()
