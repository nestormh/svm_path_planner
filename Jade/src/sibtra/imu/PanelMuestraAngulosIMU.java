/**
 * 
 */
package sibtra.imu;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;

import sibtra.gps.GPSData;
import sibtra.util.LabelDato;
import sibtra.util.LabelDatoFormato;

/**
 * @author alberto
 *
 */
public class PanelMuestraAngulosIMU extends JPanel implements IMUEventListener {
	

	private Font Grande;
	private Border blackline = BorderFactory.createLineBorder(Color.black);
	private JPanel jpCentro;
	private Vector<LabelDato> vecLabels=new Vector<LabelDato>();

	public PanelMuestraAngulosIMU() {
		setLayout(new GridLayout(0,3)); //empezamos con 3 columnas
		jpCentro=this; //no añadimos panel central
		//roll
		LabelDato lda=new LabelDatoFormato("??:??:??.??",AngulosIMU.class,"getRoll","%+10.4f");
		Grande = lda.getFont().deriveFont(20.0f);
		añadeLabelDatos(lda,"Roll");

		//Pitch
		añadeLabelDatos(new LabelDatoFormato("+???.????",AngulosIMU.class,"getPitch","%+10.4f")
		, "Pitch");
		//Yaw
		añadeLabelDatos(new LabelDatoFormato("+???.????",AngulosIMU.class,"getYaw","%+10.4f")
		, "Yaw");
		//contador
		añadeLabelDatos(new LabelDatoFormato("+???.????",AngulosIMU.class,"getContador","%7d")
		, "Contador");
	}

	/**
	 * Funcion para añadir etiqueta con todas las configuraciones por defecto
	 * @param lda etiqueta a añadir
	 * @param Titulo titulo adjunto
	 */
	private void añadeLabelDatos(LabelDato lda,String Titulo) {
		vecLabels.add(lda);
		lda.setBorder(BorderFactory.createTitledBorder(
				blackline, Titulo));
		lda.setFont(Grande);
		lda.setHorizontalAlignment(JLabel.CENTER);
		lda.setEnabled(false);
		jpCentro.add(lda);
		
	}

	/**
	 * Acciones a tomar cuando llega nuevo dato
	 * @param ang nuevo angulo encontrado
	 */
	public void actualizaAngulo(AngulosIMU ang) {
		boolean hayDato=(ang!=null);
		//atualizamos etiquetas en array
		for(int i=0; i<vecLabels.size(); i++)
			vecLabels.elementAt(i).Actualiza(ang,hayDato);
		//programamos la actualizacion de la ventana
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();
			}
		});

	}

	/* (non-Javadoc)
	 * @see sibtra.imu.IMUEventListener#handleIMUEvent(sibtra.imu.IMUEvent)
	 */
	public void handleIMUEvent(IMUEvent ev) {
		if(ev!=null)
			actualizaAngulo(ev.getAngulos());
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
