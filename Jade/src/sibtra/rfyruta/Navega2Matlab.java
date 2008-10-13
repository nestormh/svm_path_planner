package sibtra.rfyruta;

import java.awt.BorderLayout;
import java.awt.Dimension;
import javax.swing.JFrame;

import sibtra.lms.BarridoAngular;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS;


/**
 * Para realizar la detección de obstáculos con el RF a partir de los datos pasados
 * desde Matlab
 * @author alberto
 *
 */
public class Navega2Matlab {

	private ManejaLMS manLMS;

	double[][] Tr=null;
	private MiraObstaculo mi;
	private JFrame ventanaPMOS;
	private JFrame ventanaPMO;
	private PanelMiraObstaculo pmo;
	private PanelMiraObstaculoSubjetivo PMOS;

	public Navega2Matlab(String puertoRF, double[][] TrSeguir) {

		if(TrSeguir==null) {
			System.err.println("Necesaria ruta a seguir");
		}
		Tr=TrSeguir;

		//Conectamos a RF
		try { 		
			manLMS=new ManejaLMS(puertoRF);
			manLMS.setDistanciaMaxima(80);
			manLMS.CambiaAModo25(); 
		} catch (LMSException e) {
			System.err.println("No fue posible conectar o configurar RF");
		}

		System.out.println("Longitud de la trayectoria="+Tr.length);

		mi=new MiraObstaculo(Tr);
		try {
			PMOS=new PanelMiraObstaculoSubjetivo(mi,(short)manLMS.getDistanciaMaxima());
		} catch (LMSException e) {
			System.err.println("Problema al obtener distancia maxima configurada");
			System.exit(1);
		}

		ventanaPMOS=new JFrame("Mira Obstáculo Subjetivo");
		ventanaPMOS.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventanaPMOS.getContentPane().add(PMOS,BorderLayout.CENTER);
		ventanaPMOS.setSize(new Dimension(800,400));
		ventanaPMOS.setVisible(true);


		ventanaPMO=new JFrame("Mira Obstáculo");
		ventanaPMO.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pmo=new PanelMiraObstaculo(mi);
		ventanaPMO.getContentPane().add(pmo,BorderLayout.CENTER);
		ventanaPMO.setSize(new Dimension(800,600));
		ventanaPMO.setVisible(true);

	}

	/**
	 * En un momento dado nos dice a que distancia se encuentra el obstaculo más cercano
	 * @param posicionLocal Posición en coordenadas locales donde nos encontramos
	 * @param yawA rumbo actual del vehiculo hacia el norte (EN RADIANES)
	 * @return distancia al obstáculo más cercano.
	 */
	public double masCercano(double[] posicionLocal, double yawA) {
		if(posicionLocal==null || posicionLocal.length<2) {
			System.err.println("La posición local no es vector de dos posiciones");
		}
		// pedimos barrido y miramos si hay obstáculo
		//Damos pto, orientación y barrido
		double dist=Double.NaN;
		try {
			manLMS.pideBarrido((short)0, (short)180, (short)1);
			BarridoAngular ba=manLMS.recibeBarrido();
			double[] ptoAct={posicionLocal[0], posicionLocal[1]};
			dist=mi.masCercano(ptoAct, yawA, ba);
			pmo.actualiza();
			PMOS.actualiza();
			if(Double.isInfinite(dist))
				System.out.println("Estamos fuera del camino");
			else
				System.out.println("Distancia="+dist);

//			System.out.println(" iAD="+PMOS.MI.iAD
//			+"\n iAI="+PMOS.MI.iAI
//			+"\n iptoD ="+PMOS.MI.iptoD
//			+" \n iptoI ="+PMOS.MI.iptoI
//			+" \n iptoDini ="+PMOS.MI.iptoDini
//			+" \n iptoIini ="+PMOS.MI.iptoIini
//			+" \n imin ="+PMOS.MI.indMin
//			);
		} catch (LMSException e) {
			System.err.println("Problemas al obtener barrido en el punto ("
					+posicionLocal[0]
					               +","
					               +posicionLocal[1]
					                              +") :"+e.getMessage()
			);
		}
		return dist;
	}
}
