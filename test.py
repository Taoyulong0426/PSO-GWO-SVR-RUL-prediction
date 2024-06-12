from pso import PSO

if __name__ == '__main__':
    NGEN = 20
    popsize = 20
    low = [0,0]
    up = [10,100]
    parameters = [NGEN, popsize, low, up]
    pso = PSO(parameters)
    pso.main()
