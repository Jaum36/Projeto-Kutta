import numpy as np
from sympy import symbols, Function, sympify, lambdify
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt

def runge_kutta_4th_order(f, x0, y0, x_max, n):
    try:
        h = (x_max - x0) / n
        x_values = np.linspace(x0, x_max, n+1)
        y_values = np.zeros(n+1)
        y_values[0] = y0

        for i in range(n):
            x = x_values[i]
            y = y_values[i]
            k1 = h * f(x, y)
            k2 = h * f(x + h/2, y + k1/2)
            k3 = h * f(x + h/2, y + k2/2)
            k4 = h * f(x + h, y + k3)
            y_values[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

        return x_values, y_values
    except ValueError or TypeError or SyntaxError:
        messagebox.showerror("ERRO", "Invalido")

def runge_kutta_2nd_order(f, x0, y0, x_max, n):
    h = (x_max - x0)/n
    x_values = np.linspace(x0, x_max,n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(n):
        x = x_values[i]
        y = y_values[i]
        k1 = h * f(x,y)
        k2 = h * f(x + h, y + k1)
        y_values[i+1] = y + (h/2)*(k1+k2)
    
    return x_values, y_values

def runge_kutta_6th_order(f, x0, y0, x_max, n):
    h =(x_max- x0)/n
    x_values = np.linspace(x0, x_max, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(n):
        x = x_values[i]
        y = y_values[i]
        k1 = h * f(x,y)
        k2 = h * f(x + h/3, y + k1/3)
        k3 = h * f(x + (2*h/3), y + 2*k2/3)
        k4 = h * f(x + h/3, y + k1/12 + k2/3 - k3/12)
        k5 = h * f(x + h/2, y - k1/12 + (9*k2/8) - (3*k3/16) - (3*k4/8))
        k6 = h * f(x + h/2, y - (9*k2/8) - (3*k3/8) - (3*k4/4) + k5/2)
        k7 = h * f(x + h, y + (9*k1/44) - (9*k2/11) + (63*k3/44) + (18*k4/11) - (16*k6/11))
        y_values[i+1] = y + ((11*k1/120) + (27*k3/40) + (27*k4/40) - (4*k5/15) - (4*k6/15) + (11*k7/120))

    return x_values, y_values

def euler(f, x0, y0, x_max, n):
    h = (x_max - x0)/n
    x_values = np.linspace(x0,x_max,n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(n):
        x = x_values[i]
        y = y_values[i]
        y_values[i+1] = y + h*f(x,y)

    return x_values, y_values

def heun(f, x0, y0, x_max, n):
    h = (x_max - x0)/n
    x_values = np.linspace(x0, x_max, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(n):
        x = x_values[i]
        y = y_values[i]
        y_values[i+1] = y + (1/2) * (f(x,y)+f(x+h, y + h*f(x,y)))

    return x_values, y_values

def solve_edo1(equation_str, x0, y0, x_final, n, method):
    try:
        x, y = symbols('x y')
        derivative_expr = sympify(equation_str)
        f = lambdify((x, y), derivative_expr, modules=['numpy'])

        if method == 'RK4':
            return runge_kutta_4th_order(f, x0, y0, x_final, n)
        elif method == 'RK2':
            return runge_kutta_2nd_order(f, x0, y0, x_final, n)
        elif method == 'RK6':
            return runge_kutta_6th_order(f, x0, y0, x_final, n)
        elif method == 'Euler':
            return euler(f, x0, y0, x_final, n)
        elif method == 'Heun':
            return heun(f, x0, y0, x_final, n)
        else:
            raise ValueError("Método desconhecido")
    except ValueError or TypeError or SyntaxError:
        messagebox.showerror("Error", "Invalido")
    

def plot_graph(x_values, y_values):
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, label="y(x)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gráfico de y em função de x")
        plt.legend()
        plt.grid(True)
        plt.show()

def run():
        equation_str = equation_entry.get()
        x0 = float(x0_entry.get())
        y0 = float(y0_entry.get())
        x_final = float(x_final_entry.get())
        n = int(n_entry.get())
        method = method_var.get()

        x_values, y_values = solve_edo1(equation_str, x0, y0, x_final, n, method)
        plot_graph(x_values, y_values)


root = tk.Tk()
root.title("Método de Runge-Kutta para EDOs")


tk.Label(root, text="Digite a derivada de y (y'):").pack()
equation_entry = tk.Entry(root)
equation_entry.pack()

tk.Label(root, text="x0:").pack()
x0_entry = tk.Entry(root)
x0_entry.pack()

tk.Label(root, text="y0:").pack()
y0_entry = tk.Entry(root)
y0_entry.pack()

tk.Label(root, text="x final:").pack()
x_final_entry = tk.Entry(root)
x_final_entry.pack()

tk.Label(root, text="Número de pontos (n):").pack()
n_entry = tk.Entry(root)
n_entry.pack()


method_var = tk.StringVar()

R4 = tk.Radiobutton(root, text="Runge-Kutta 4ª Ordem", variable=method_var, value='RK4')
R2 = tk.Radiobutton(root, text="Runge-Kutta 2ª Ordem", variable=method_var, value='RK2')
R6 = tk.Radiobutton(root, text = "Runge-Kutta 6ª Ordem", variable=method_var, value='RK6')
E = tk.Radiobutton(root, text="Método de Euler", variable=method_var, value='Euler')
H = tk.Radiobutton(root, text="Método de Heun", variable=method_var, value='Heun')

R2.pack()
R4.pack()
R6.pack()
E.pack()
H.pack()


tk.Button(root, text="Resolver e Plotar", command=run).pack()

root.mainloop()
