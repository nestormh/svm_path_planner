/*
 * Creado el 22/02/2008
 *
 * Creado por Alberto Hamilcon con Eclipse
 */
package sibtra.lms;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Arc2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.Random;

import javax.swing.Box;
import javax.swing.JCheckBox;
import javax.swing.JFrame;

import sibtra.lms.BarridoAngular;
import sibtra.lms.ZonaLMS;
import sibtra.lms.ZonaRadialLMS;
import sibtra.lms.ZonaRectangularLMS;
import sibtra.lms.ZonaSegmentadaLMS;
import sibtra.lms.BarridoAngular.barridoAngularIterator;
import sibtra.lms.ZonaSegmentadaLMS.pointIterator;
import sibtra.util.PanelBarrido;

@SuppressWarnings("serial")
public class PanelMuestraBarrido extends PanelBarrido {
		
	private JCheckBox jcbBarrido;
	private JCheckBox jcbZonaA;
	private JCheckBox jcbZonaB;
	private JCheckBox jcbZonaC;
		
	/**
	 * Barrido actual
	 */
	private BarridoAngular barridoAct=null;
	
	
	private ZonaLMS	Zona1A=null;
	private ZonaLMS	Zona1B=null;
	private ZonaLMS	Zona1C=null;
	private ZonaLMS	Zona2A=null;
	private ZonaLMS	Zona2B=null;
	private ZonaLMS	Zona2C=null;


	/**
	 * Crea parte grafica junto con slider de zoom
	 * @param distanciaMaxima Distancia máxima del gráfico
	 */
	public PanelMuestraBarrido(short distanciaMaxima) {
		super(distanciaMaxima);
			
		{  //añadimos los checkbox que necesito
			
			jpChecks.add(Box.createHorizontalStrut(15));

			jcbBarrido=new JCheckBox("Barrido",true);
			jcbBarrido.addChangeListener(this);
			jpChecks.add(jcbBarrido);
			jcbBarrido.setEnabled(false);
			
			jpChecks.add(Box.createHorizontalStrut(15));
			
			jcbZonaA=new JCheckBox("Zona A",true);
			jcbZonaA.addChangeListener(this);
			jpChecks.add(jcbZonaA);
			jcbZonaA.setEnabled(false);
			
			jpChecks.add(Box.createHorizontalStrut(15));
			
			jcbZonaB=new JCheckBox("Zona B",true);
			jcbZonaB.addChangeListener(this);
			jpChecks.add(jcbZonaB);
			jcbZonaB.setEnabled(false);
			
			jpChecks.add(Box.createHorizontalStrut(15));
			
			jcbZonaC=new JCheckBox("Zona C",true);
			jcbZonaC.addChangeListener(this);
			jpChecks.add(jcbZonaC);
			jcbZonaC.setEnabled(false);

			jpChecks.add(Box.createHorizontalStrut(15));
			
		}

	}

	protected void cosasAPintar(Graphics g0) {
		super.cosasAPintar(g0);
		Graphics2D g=(Graphics2D)g0;

		if(barridoAct!=null && jcbBarrido.isSelected()) {
			//pasamos a representar el barrido
			g.setColor(Color.GREEN);
			g.setStroke(new BasicStroke());
			GeneralPath perimetro = 
				new GeneralPath(GeneralPath.WIND_EVEN_ODD, barridoAct.numDatos());

			barridoAngularIterator baIt=barridoAct.creaIterator();
			baIt.next(); //para obtener el primer punto
			Point2D.Double px=point2Pixel(baIt.punto());
			perimetro.moveTo((float)px.getX(),(float)px.getY());
			while(baIt.next()) {
				px=point2Pixel(baIt.punto());
				//Siguientes puntos son lineas
				perimetro.lineTo((float)px.getX(),(float)px.getY());
			}
			g.draw(perimetro);
		}

		g.setStroke(new BasicStroke());
		if(Zona1A!=null && jcbZonaA.isSelected()) {
			g.setColor(Color.YELLOW);
			pintaZona(Zona1A, g);
		}
		if(Zona1B!=null && jcbZonaB.isSelected()) {
			g.setColor(Color.BLUE);
			pintaZona(Zona1B, g);
		}				
		if(Zona1C!=null && jcbZonaC.isSelected()) {
			g.setColor(Color.ORANGE);
			pintaZona(Zona1C, g);
		}
	}

	private void pintaZona(ZonaLMS za, Graphics2D g) {
		//Tenemos que ver que tipo de zona es para representarla
		if(ZonaRadialLMS.class.isInstance(za)) {
			//es zona radial
			ZonaRadialLMS zr=(ZonaRadialLMS)za;
			double radio=zr.radioZona/(zr.isEnMilimetros()?1000.0:100.0);
			Point2D.Double pxEsqSI=point2Pixel((double)-radio,(double)radio)
			,pxEsqID=point2Pixel((double)+radio,(double)-radio);
			g.draw(new Arc2D.Double(pxEsqSI.getX(),pxEsqSI.getY()
					,pxEsqID.getX()-pxEsqSI.getX(),pxEsqID.getY()-pxEsqSI.getY()
					,0.0, 180.0
					,Arc2D.OPEN));
		} else if (ZonaRectangularLMS.class.isInstance(za)) {
			//es zona rectangular
			ZonaRectangularLMS zr=(ZonaRectangularLMS)za;
			//Calculamos las esquinas del rectangulo
			Point2D.Double pxEsqSI=point2Pixel((double)-zr.distanciaIzda/(zr.isEnMilimetros()?1000.0:100.0)
					,(double)zr.distanciaFrente/(zr.isEnMilimetros()?1000.0:100.0));
			Point2D.Double pxEsqID=point2Pixel((double)zr.distanciaDecha/(zr.isEnMilimetros()?1000.0:100.0),(double)0);
			//pintamos el rectángulo
			g.draw(new Rectangle2D.Double(pxEsqSI.getX(), pxEsqSI.getY()
					,pxEsqID.getX()-pxEsqSI.getX()
					,pxEsqID.getY()-pxEsqSI.getY())
			);
		} else if (ZonaSegmentadaLMS.class.isInstance(za) ) {
			//es zona segmentada
			ZonaSegmentadaLMS zs=(ZonaSegmentadaLMS)za;
			if(zs.radiosPuntos!=null && zs.radiosPuntos.length>0) {
				GeneralPath perimetro = 
					new GeneralPath(GeneralPath.WIND_EVEN_ODD, zs.radiosPuntos.length);
				pointIterator pi=zs.creaPointIterator();
				//Punto inicial del camino
				Point2D.Double px=point2Pixel(pi.next());
				perimetro.moveTo((float)px.getX(),(float)px.getY());
				while(pi.hasNext()) {
					px=point2Pixel(pi.next());
					perimetro.lineTo((float)px.getX(),(float)px.getY());
				}
				g.draw(perimetro);
			}
		}

	}
	
	/**
	 * Para cambiar el barrido que se está mostrando.
	 * NO actualiza la presentación, sólo cambia los datos.
	 * @param barr barrido nuevo
	 */
	public void setBarrido(BarridoAngular barr) {
		//TODO seguramente es más correcto copiarlo
		barridoAct=barr;
		
		if(barr==null) {
			jcbBarrido.setEnabled(false);
			return;
		}
		jcbBarrido.setEnabled(true);
		//cambiamos color de nombres de las zonas si están infringidas
		if(barr.infringeA())
			jcbZonaA.setForeground(Color.RED);
		else
			jcbZonaA.setForeground(Color.BLACK);
		if(barr.infringeB())
			jcbZonaB.setForeground(Color.RED);
		else
			jcbZonaB.setForeground(Color.BLACK);
		if(barr.infringeC())
			jcbZonaC.setForeground(Color.RED);
		else
			jcbZonaC.setForeground(Color.BLACK);
	}
	/**
	 * Devuelve la zona del conjunto especificado
	 * @param del1 si es del 1
	 * @param queZona cual zona (A, B ó C)
	 * @return null si no es id de zona
	 */
	public ZonaLMS getZona(boolean del1, byte queZona) {
		if(del1) {
			//se quiere la del 1
			if(queZona==ZonaLMS.ZONA_A)	return Zona1A;
			if(queZona==ZonaLMS.ZONA_B) return Zona1B;
			if(queZona==ZonaLMS.ZONA_C)	return Zona1C;
		} else {
			//se quiere la del 2
			if(queZona==ZonaLMS.ZONA_A)	return Zona2A;
			if(queZona==ZonaLMS.ZONA_B) return Zona2B;
			if(queZona==ZonaLMS.ZONA_C)	return Zona2C;
		}
		return null;
	}

	/**
	 * Establece una zona. Se mira el conjnto y la zona para sustituir la que se tiene.
	 * NO actualiza la presentación.
	 * @param zona a actualizar
	 */
	public void setZona(ZonaLMS zona) {
		if (zona==null) return;
		if(zona.isConjunto1()){
			//es del conjunto 1
			if(zona.getQueZona()==ZonaLMS.ZONA_A) { 
				Zona1A=zona;
				jcbZonaA.setEnabled(true);
			}
			if(zona.getQueZona()==ZonaLMS.ZONA_B) {
				Zona1B=zona;
				jcbZonaB.setEnabled(true);
			}
			if(zona.getQueZona()==ZonaLMS.ZONA_C) {
				Zona1C=zona;
				jcbZonaC.setEnabled(true);
			}
		} else {
			if(zona.getQueZona()==ZonaLMS.ZONA_A) {
				Zona2A=zona;
				jcbZonaA.setEnabled(true);
			}
			if(zona.getQueZona()==ZonaLMS.ZONA_B) {
				Zona2B=zona;
				jcbZonaB.setEnabled(true);
			}
			if(zona.getQueZona()==ZonaLMS.ZONA_C) {
				Zona2C=zona;
				jcbZonaC.setEnabled(true);
			}
		}
		repaint();
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		JFrame VentanaPrincipal=new JFrame("PanelMuestraBarrido");
		
		PanelMuestraBarrido PMB=new PanelMuestraBarrido((short)80);
		
		VentanaPrincipal.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		VentanaPrincipal.add(PMB);
		VentanaPrincipal.setSize(new Dimension(800,400));
		VentanaPrincipal.setVisible(true);
		
		//generamos barrido aleatorio en de resolucion grado
		BarridoAngular barr=new BarridoAngular(181,0,4,(byte)0,false,(short)2); //crea los arrays de 181 datos.
		Random ran=new Random();
		double nd; 
		for(int i=0; i<=180; i++) {
			do {
				nd=(40+20*ran.nextGaussian());
			} while(nd>80 || nd<=0);

			//nd=40;
			barr.datos[i]=(short)(nd*100);
		}
		
		PMB.setBarrido(barr);
		
		PMB.setZona(new ZonaRadialLMS((short)180,(short)50,true,true,ZonaLMS.ZONA_A,(short)25000));
		
		PMB.setZona(new ZonaRectangularLMS((short)180,(short)50,true,true,ZonaLMS.ZONA_B
				,(short)20000,(short)15000,(short)30000));

		ZonaSegmentadaLMS zs=new ZonaSegmentadaLMS((short)180,(short)100,false,true,ZonaLMS.ZONA_C,(short)30);
		for(int i=0; i<=30; i++)
			zs.radiosPuntos[i]=990;
		zs.radiosPuntos[0]=0;
		zs.radiosPuntos[30]=0;
		PMB.setZona(zs);
		
		PMB.actualiza();
		
	}


}
