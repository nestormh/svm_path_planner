package sibtra.rfycarro;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Arc2D;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.lms.BarridoAngular;
import sibtra.lms.PanelMuestraBarrido;
import sibtra.util.Parametros;

/**
 * Panel para mostrar la información de @link {@link FuturoObstaculo}
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class PanelFuturoObstaculo extends PanelMuestraBarrido {
	
	FuturoObstaculo futObs;
	/** Tamaño de la marca del punto más cercano */
	double TamCruz=10;
	
	public PanelFuturoObstaculo(FuturoObstaculo fo) {
		super((short) 80);
		futObs=fo;
		
	}

	protected void cosasAPintar(Graphics g0) {
		super.cosasAPintar(g0);
		Graphics2D g=(Graphics2D)g0;
		
		if(futObs==null) return;
		g.setColor(Color.BLUE);
		//pintamos por donde vamos a estar
		if(Math.abs(futObs.alfaAct)>Math.toRadians(0.1)) {
			//Pintamos los circulos derechos e izquierdo
			Point2D.Double prR=new Point2D.Double(-futObs.radioCur*futObs.signoAlfa
					,-Parametros.distRFEje*0.9);
			g.draw(arco(prR,futObs.radioExterior,0,180));
			g.draw(arco(prR,futObs.radioInterior,0,180));
		} else {
			//pintamos rectas
			//izquierda
			Point2D.Double pxIzda=point2Pixel(-Parametros.medioAnchoCarro, 0);
			g.draw(new Line2D.Double(pxIzda.x,0,pxIzda.x,pxIzda.y));
			//derecha
			Point2D.Double pxDecha=point2Pixel(+Parametros.medioAnchoCarro, 0);
			g.draw(new Line2D.Double(pxDecha.x,0,pxDecha.x,pxDecha.y));
			
		}
		//Marcamos en punto más cercano si existe
		if(futObs.indMin>0) {
		g.setColor(Color.RED);
		Point2D.Double pxCercano=point2Pixel(futObs.bAct.getPunto(futObs.indMin));
		g.draw(new Line2D.Double(pxCercano.x-TamCruz, pxCercano.y-TamCruz 
				,pxCercano.x+TamCruz, pxCercano.y+TamCruz));
		g.draw(new Line2D.Double(pxCercano.x-TamCruz, pxCercano.y+TamCruz 
				,pxCercano.x+TamCruz, pxCercano.y-TamCruz));
		}

	}
	
	/** Arco dado el centro y el radio */
	private Arc2D.Double arco(Point2D.Double centro, double radio, double angIni, double grados) {
		Point2D.Double extremo=new Point2D.Double(centro.x-radio,centro.y+radio);
		Point2D.Double pxCentro=point2Pixel(centro);
		Point2D.Double pxExtremo=point2Pixel(extremo);
		double w=(pxCentro.x-pxExtremo.x)*2;
		double h=(pxCentro.y-pxExtremo.y)*2;
		return new Arc2D.Double(pxExtremo.x,pxExtremo.y
				,w
				,h
				,angIni,grados
				,Arc2D.OPEN
				);
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		final FuturoObstaculo fo=new FuturoObstaculo();
		final SpinnerNumberModel spnAlfa;
		final SpinnerNumberModel spnVel;
		final BarridoAngular ba;
		//Damos pto, orientación y barrido
		ba=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
		for(int i=0;i<ba.numDatos();i++) {
//			ba.datos[i]=(short)((15.0)*100.0);
			ba.datos[i]=(short)((Math.sin((double)i/(ba.numDatos()-1)*Math.PI*13.6)*3.0+10.0)*100.0);
		}
		
		JFrame ventana=new JFrame("Panel Futuro Obstáculo");
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		final PanelFuturoObstaculo pfo=new PanelFuturoObstaculo(fo);
		ventana.add(pfo);
		
		{ //panel inferior para variar alfa y velocidad
			JPanel jpSur=new JPanel();
			
			spnAlfa=new SpinnerNumberModel(10,-45,45,0.1);
			JSpinner jsAlfa=new JSpinner(spnAlfa);
			jpSur.add(jsAlfa);
			
			spnVel=new SpinnerNumberModel(1,0,6,0.5);
			JSpinner jsVel=new JSpinner(spnVel);
			jpSur.add(jsVel);
			final JLabel jlTiempo;
			jlTiempo=new JLabel("Tiempo= ###.## sg");
			jpSur.add(jlTiempo);
			
			ChangeListener chL=new ChangeListener() {
				public void stateChanged(ChangeEvent arg0) {
					double alfa=Math.toRadians(spnAlfa.getNumber().doubleValue());
					double velMS=spnVel.getNumber().doubleValue();
					double tiempo=fo.tiempoAObstaculo(alfa, velMS, ba);
					jlTiempo.setText(String.format("Tiempo= %5.2f sg", tiempo));
					pfo.actualiza();
				}
			};

			spnAlfa.addChangeListener(chL);
			spnVel.addChangeListener(chL);
			
			ventana.getContentPane().add(jpSur, BorderLayout.SOUTH);
			
		}
			
		
		ventana.setSize(new Dimension(800,600));
		ventana.setVisible(true);
		
		double tiempo=fo.tiempoAObstaculo(Math.toRadians(10), 1, ba);
		pfo.setBarrido(ba);
		pfo.actualiza();

	}

}
