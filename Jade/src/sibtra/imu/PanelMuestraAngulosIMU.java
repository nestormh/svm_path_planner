/**
 * 
 */
package sibtra.imu;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;

/**
 * @author alberto
 *
 */
public class PanelMuestraAngulosIMU extends JPanel implements IMUEventListener {
	
	private JLabel jlRoll;
	private JLabel jlPitch;
	private JLabel jlYaw;
	private JLabel jlContador;

	public PanelMuestraAngulosIMU() {
		setLayout(new GridLayout(0,3)); //empezamos con 3 columnas
		Border blackline = BorderFactory.createLineBorder(Color.black);
		Font Grande;
		JLabel jla; //variable para poner JLable actual

		{//roll
			jlRoll=jla=new JLabel("+???.????");
		    Grande = jla.getFont().deriveFont(20.0f);
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Roll"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}

		{//Pitch
			jlPitch=jla=new JLabel("+???.????");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Pitch"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}
	
		{//Yaw
			jlYaw=jla=new JLabel("+???.????");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Yaw"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}
		{//contador
			jlContador=jla=new JLabel("?????");
			jla.setBorder(BorderFactory.createTitledBorder(
					       blackline, "Contador"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			add(jla);
		}
	}
	
	public void actualizaAngulo(AngulosIMU ang) {
		if(ang==null) {
			jlRoll.setEnabled(false);
			jlPitch.setEnabled(false);
			jlYaw.setEnabled(false);
			jlContador.setEnabled(false);
		} else {
			jlRoll.setEnabled(true);
			jlPitch.setEnabled(true);
			jlYaw.setEnabled(true);
			jlContador.setEnabled(true);
			//nuevos valores
			jlRoll.setText(String.format("%+10.4f", ang.roll));
			jlPitch.setText(String.format("%+10.4f", ang.pitch));
			jlYaw.setText(String.format("%+10.4f", ang.yaw));
			jlContador.setText(String.format("%7d", ang.contador));
		}
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
		pmai.actualizaAngulo(new AngulosIMU(0,0,0,0));
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
