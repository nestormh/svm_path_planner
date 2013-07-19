package sibtra.odometria;

import javax.swing.JFrame;

public class PanelOdometria extends PanelMuestraOdometria {
	
	ConexionSerialOdometria conexionOdo;
	
	public PanelOdometria(ConexionSerialOdometria conexOdo) {
		super(); //solo espaciales
		if(conexOdo==null)
			throw new IllegalArgumentException("Conexion a IMU no puede ser null");
		this.conexionOdo=conexOdo;
	}
	
	/** En la actulización periodica usamos el último ángulo */
	protected void actualiza() {
//		actualizaAngulo(conexionOdo.get());
		super.actualiza();
	}
	
	public static void main(String[] args) {
		JFrame ventanaPrincipal=new JFrame("Panel Muestra Odometria");
		ConexionSerialOdometria csOdo=new ConexionSerialOdometria("/dev/ttyUSB0");
		PanelOdometria pmOdo=new PanelOdometria(csOdo);
		DatosOdometria datos = new DatosOdometria(0,0,0);
		pmOdo.actualizaDatosOdometria(datos);
		ventanaPrincipal.add(pmOdo);		
		csOdo.addOdometriaEventListener(pmOdo);
		ventanaPrincipal.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventanaPrincipal.pack();
		ventanaPrincipal.setVisible(true);
//		pmOdo.handleOdometriaEvent(ev);
	}

}
