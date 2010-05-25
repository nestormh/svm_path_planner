package sibtra.ui.modulos;

import sibtra.gps.Trayectoria;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.ModificadorTrayectoria;
import sibtra.ui.defs.Motor;

public class ModificadorSlalom implements ModificadorTrayectoria {
	private Motor motor=null;
	private Trayectoria tr=null;
	private VentanasMonitoriza ventanaMonitoriza=null;
	double[] posCono1_X;
	double[] posCono1_Z;
	double[] posCono2_X;
	double[] posCono2_Z;
	@Override
	public void actuar() {
		// TODO Auto-generated method stub

	}

	@Override
	public void parar() {
		// TODO Auto-generated method stub

	}
	public void setPosicionConos(double[] cono1_X,double[] cono1_Z,double[] cono2_X,double[] cono2_Z){
//		if()
//			throw new IllegalStateException("Faltan coordenadas de la serie de conos");
		for(int i=0;i<cono1_X.length;i++){
			posCono1_X[i] = cono1_X[i];
		}
	}
	@Override
	public void setTrayectoriaInicial(Trayectoria tra) {
		tr = tra;
	}

	@Override
	public void setMotor(Motor mtr) {
		motor = mtr;
	}

	@Override
	public String getDescripcion() {
		// TODO Auto-generated method stub
		return "generador de rutas para slalom";
	}

	@Override
	public String getNombre() {
		// TODO Auto-generated method stub
		return "Modificador Slalom";
	}

	@Override
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		if(ventanaMonitoriza!=null)//Comprobamos que no se haya creado antes
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		ventanaMonitoriza=ventMonitoriza;
		if(motor==null)// El motor tiene que haber sido fijado antes de realizar el setVentanasMonitoriza(ver PanelEligeModulos)
			throw new IllegalStateException("Se debe haber fijado el motor antes de fijar VentanaMonitoriza");
//		modCoche=motor.getModeloCoche(); // La lógica parece decitr que no hacen falta los datos 
		// de posición y orientación del coche para calcular la ruta de slalom
		return true;
	}

	@Override
	public void terminar() {
		// TODO Auto-generated method stub

	}

}
