import Lou_MoriZwanzig as mz
import time as tm

if __name__ == '__main__':
    start = tm.time()
    print("started")
    velocity, force, sq_displacement = mz.LoadDatafile("velocities.npy", "forces.npy", "sq_displacement_stat.npy")
    part_mass, dt,KB,T,Omega = mz.VarSetter(80.,0.005,1.,1.,0)
    print("mp started")
    list_, Ein_list = mz.main(16,4,1,part_mass,dt,KB,T,Omega,velocity,force,sq_displacement) 
    result = mz.Fixup(list_,Ein_list)
    #PickleResults("NewResultTest",result)
    print(' The bootstrapping finished in:\n',round(((tm.time()-start)/60 / 60),2) ,'hours')