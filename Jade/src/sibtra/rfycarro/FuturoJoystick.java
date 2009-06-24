package sibtra.rfycarro;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.lms.BarridoAngular;

import com.centralnexus.input.Joystick;

public abstract class FuturoJoystick {
	public final static float MinY=0.0014648885f;
	public final static float MaxY=0.005401776f;
	public final static float MinX=5.1881466E-4f;
	public final static float MaxX=0.0055543687f;
	public final static double AlfaMaximo=Math.toRadians(45);
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		final FuturoObstaculo fo=new FuturoObstaculo();
		final BarridoAngular ba;
		final JLabel jlDistancia;
		final JLabel jlXAlfa;

		ba=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
		for(int i=0;i<ba.numDatos();i++) {
//			ba.datos[i]=(short)((15.0)*100.0);
			ba.datos[i]=(short)((Math.sin((double)i/(ba.numDatos()-1)*Math.PI*13.6)*3.0+10.0)*100.0);
		}
		
		JFrame ventana=new JFrame("Futuro Joystick");
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		final PanelFuturoObstaculo pfo=new PanelFuturoObstaculo(fo);
		ventana.add(pfo);
		
		{ //panel inferior para variar alfa y velocidad
			JPanel jpSur=new JPanel();
			
			jlXAlfa=new JLabel("X= ###.##  Alfa=###.## º");
			jpSur.add(jlXAlfa);
			
			jlDistancia=new JLabel("Distancia= ###.## m Tiempo=###.## sg");
			jpSur.add(jlDistancia);
			
			
			ventana.getContentPane().add(jpSur, BorderLayout.SOUTH);
		}
			
		
		ventana.setSize(new Dimension(800,600));
		ventana.setVisible(true);

		fo.distanciaAObstaculo(Math.toRadians(10), ba);
		pfo.setBarrido(ba);
		pfo.actualiza();


        try {
			Joystick joy = Joystick.createInstance();
	        for (;;) {
	            joy.poll();
	            float x=joy.getX();
	            double alfa=-(AlfaMaximo*2/(MaxX-MinX)*(x-MinX)-AlfaMaximo);
				jlXAlfa.setText(String.format("X= %f  Alfa=%f º."
						, x,Math.toDegrees(alfa)));
				double distancia=fo.distanciaAObstaculo(alfa, ba);
				double velMS=6/(MaxY-MinY)*(MaxY-joy.getY());
				jlDistancia.setText(String.format("Distancia= %5.2f m Velocidad=%f Tiempo=%5.2f sg"
						, distancia,velMS, fo.tiempoAObstaculo(velMS)));
	    		pfo.setBarrido(ba);
	    		pfo.actualiza();
	            try {
	                Thread.sleep(200);
	            } catch(InterruptedException e) {
	                break;
	            }
	        }

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
