package sibtra.odometria;

import javax.swing.JFrame;

import sibtra.imu.ConexionSerialIMU;
import sibtra.imu.PanelMuestraAngulosIMU;
import sibtra.util.LabelDato;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.ThreadSupendible;

public class PanelMuestraOdometria extends PanelFlow implements OdometriaEventListener{
	
	private int contadorUltimo;
	protected ThreadSupendible thCiclico;
	protected ConexionSerialOdometria conSerial;
	

	public PanelMuestraOdometria(){
		super();		
		// Coordenada X
		LabelDato labelX=new LabelDatoFormato(DatosOdometria.class,"getxRel","%+10.4f");
		añadeAPanel(labelX,"Coordenada x");
		// Coordenada Y
		LabelDato labelY=new LabelDatoFormato(DatosOdometria.class,"getyRel","%+10.4f");
		añadeAPanel(labelY,"Coordenada y");
		//Orientacion
		LabelDato labelOri=new LabelDatoFormato(DatosOdometria.class,"getYaw","%+10.4f");
		añadeAPanel(labelOri,"Orientacion");
	}
	
	public PanelMuestraOdometria(ConexionSerialOdometria serial){
		super();		
		conSerial = serial;
		// Coordenada X
		LabelDato labelX=new LabelDatoFormato(DatosOdometria.class,"getxRel","%+10.4f");
		añadeAPanel(labelX,"Coordenada x");
		// Coordenada Y
		LabelDato labelY=new LabelDatoFormato(DatosOdometria.class,"getyRel","%+10.4f");
		añadeAPanel(labelY,"Coordenada y");
		//Orientacion
		LabelDato labelOri=new LabelDatoFormato(DatosOdometria.class,"getYaw","%+10.4f");
		añadeAPanel(labelOri,"Orientacion");
	}
	
	public void actualizaDatosOdometria (DatosOdometria odo){
		//si el nuevo dato tiene el mismo contador no es nuevo.
		if(odo!=null)
			if(odo.getContador()==contadorUltimo)
				odo=null;
			else 
				contadorUltimo=odo.getContador();
		actualizaDatos(odo);
	}
	/**
	 * Método que maneja el evento serial de la electrónica de la odometría. 
	 * Manda a repintar el panel cada vez que se reciba uno.
	 */
	@Override
	public void handleOdometriaEvent(OdometriaEvent ev) {
		if(ev!=null) {
			actualizaDatosOdometria(ev.getDatosOdometria());
			repinta();
		}		
	}
	public static void main(String[] args) {
		JFrame ventanaPrincipal=new JFrame("Panel Muestra Odometria");
		PanelMuestraOdometria pmOdo=new PanelMuestraOdometria();
		DatosOdometria datos = new DatosOdometria(0,0,0);
		pmOdo.actualizaDatosOdometria(datos);
		ventanaPrincipal.add(pmOdo);		
		ConexionSerialOdometria csOdo=new ConexionSerialOdometria("/dev/ttyS0");
//		pmOdo.handleOdometriaEvent(ev);

		
//		csOdo.addIMUEventListener(pmOdo);

		//ventanaPrincipal.setSize(new Dimension(800,400));
		ventanaPrincipal.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventanaPrincipal.pack();
		ventanaPrincipal.setVisible(true);
//		int cont = 0;
//		int cuentasLeft = 0;
//		int cuentasRight = 0;
//		double Ts = 0.25;
//		while (true){
////			datos.setxRel(1+cont);
////			datos.setyRel(1+cont);
////			datos.setYaw(3+cont);
//			datos.calculaDatos(cuentasLeft, cuentasRight, Ts);
//			datos.setContador(cont);
//			pmOdo.actualizaDatosOdometria(datos);
//			cont++;
//			
//			cuentasLeft = cuentasLeft + 1;
//			cuentasRight = cuentasRight + 2;
//		}
	}

}
