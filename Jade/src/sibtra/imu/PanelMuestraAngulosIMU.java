/**
 * 
 */
package sibtra.imu;

import java.awt.GridLayout;

import javax.swing.JFrame;

import sibtra.util.LabelDato;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelDatos;

/**
 * Mostrar los 3 angulos y el contador de recepción de la IMU
 * @author alberto
 *
 */
public class PanelMuestraAngulosIMU extends PanelDatos implements IMUEventListener {
	
	/**	Contador del último dato presentado */
	private int contadorUltimo=0;

	public PanelMuestraAngulosIMU() {
		super();
		setLayout(new GridLayout(0,2)); //empezamos con 3 columnas
		//roll
		LabelDato lda=new LabelDatoFormato("??:??:??.??",AngulosIMU.class,"getRoll","%+10.4f");
		añadeAPanel(lda,"Roll");

		//Pitch
		añadeAPanel(new LabelDatoFormato("+???.????",AngulosIMU.class,"getPitch","%+10.4f")
		, "Pitch");
		//Yaw
		añadeAPanel(new LabelDatoFormato("+???.????",AngulosIMU.class,"getYaw","%+10.4f")
		, "Yaw");
		//contador
		añadeAPanel(new LabelDatoFormato("+???.????",AngulosIMU.class,"getContador","%7d")
		, "Contador");
	}


	/**
	 * Acciones a tomar cuando llega nuevo dato
	 * @param ang nuevo angulo encontrado
	 */
	public void actualizaAngulo(AngulosIMU ang) {
		//si el nuevo dato tiene el mismo contador no es nuevo.
		if(ang!=null)
			if(ang.getContador()==contadorUltimo)
				ang=null;
			else 
				contadorUltimo=ang.getContador();
		actualizaDatos(ang);
	}

	/* (non-Javadoc)
	 * @see sibtra.imu.IMUEventListener#handleIMUEvent(sibtra.imu.IMUEvent)
	 */
	public void handleIMUEvent(IMUEvent ev) {
		if(ev!=null) {
			actualizaAngulo(ev.getAngulos());
			repinta();
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		JFrame ventanaPrincipal=new JFrame("Panel Muestra Angulos");
		PanelMuestraAngulosIMU pmai=new PanelMuestraAngulosIMU();
		pmai.actualizaAngulo(null);
		ventanaPrincipal.add(pmai);
		
		ConexionSerialIMU csi=new ConexionSerialIMU();
		csi.ConectaPuerto("/dev/ttyUSB0");
		
		csi.addIMUEventListener(pmai);

		//ventanaPrincipal.setSize(new Dimension(800,400));
		ventanaPrincipal.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventanaPrincipal.pack();
		ventanaPrincipal.setVisible(true);

	}

}
