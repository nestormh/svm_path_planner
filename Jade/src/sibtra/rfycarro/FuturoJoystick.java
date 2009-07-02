package sibtra.rfycarro;

import java.awt.BorderLayout;
import java.awt.Dimension;

import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import sibtra.lms.BarridoAngular;
import sibtra.util.ManejaJoystick;
import sibtra.util.PanelJoystick;

public abstract class FuturoJoystick {
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
		JPanel central=new JPanel();
		ventana.add(central);
		central.setLayout(new BoxLayout(central,BoxLayout.PAGE_AXIS));
		central.add(pfo);
		
		ManejaJoystick joy = new ManejaJoystick();
		PanelJoystick panJ=new PanelJoystick(joy);
		central.add(panJ);
		
		
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
		panJ.actualiza();


		for (;;) {
			panJ.actualiza();
			jlXAlfa.setText(String.format("X= %f  Alfa=%f º."
					, joy.getX(),Math.toDegrees(joy.getAlfa())));
			double distancia=fo.distanciaAObstaculo(joy.getAlfa(), ba);
			jlDistancia.setText(String.format("Distancia= %5.2f m Velocidad=%f Tiempo=%5.2f sg Avance=%5.2f Y=%f"
					, distancia,joy.getVelocidad(), fo.tiempoAObstaculo(joy.getVelocidad())
					,joy.getAvance()
					,joy.getY()));
			pfo.setBarrido(ba);
			pfo.actualiza();
			
			try {
				Thread.sleep(200);
			} catch(InterruptedException e) {
				break;
			}
		}


	}

}
