/*
 * Creado el 22/02/2008
 *
 * Creado por Alberto Hamilcon con Eclipse
 */
package sibtra.util;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Arc2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.geom.Point2D.Double;
import java.util.Random;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingConstants;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.lms.BarridoAngular;
import sibtra.lms.ZonaLMS;
import sibtra.lms.ZonaRadialLMS;
import sibtra.lms.ZonaRectangularLMS;
import sibtra.lms.ZonaSegmentadaLMS;
import sibtra.lms.BarridoAngular.barridoAngularIterator;
import sibtra.lms.ZonaSegmentadaLMS.pointIterator;

public class PanelBarrido extends JPanel implements ChangeListener, MouseListener {
	
	protected static final int TamMarca = 40;

	/**
	 * Para variar zoom
	 */
	private JSlider JSliderZoom;
	
	/**
	 * Panel donde pintar
	 */
	private JPanel	JPanelGrafico;
		
	private JCheckBox jcbRegla;
	private JCheckBox jcbRejillaRadial;
	private JCheckBox jcbRejillaRecta;

	//Abajo los checks para mostrar o no las zonas y el barrido
	protected JPanel jpChecks;
	
	/**
	 * Distancia maxima a representar
	 */
	private short distMax;
		
	/**
	 * Coordenadas de la esquina superior izquierda.
	 * En unidades mundo real.
	 */
	private Point2D esqSI;
	
	/**
	 * Coordenadas de la esquina superior izquierda.
	 * En unidades mundo real.
	 */
	private Point2D esqID;

	private MouseEvent evenPulsa;

	private boolean restaurar;
	
	/** Distancia (del mundo real) más grande que se ve en panel */
	protected double distanciaVista;


	/**
	 * Convierte punto en el mundo real a punto en la pantalla.
	 * @param pt punto del mundo real
	 * @return punto en pantalla
	 */
	protected Point2D.Double point2Pixel(Point2D pt) {
		return new Point2D.Double(
				(pt.getX()-esqSI.getX())*JPanelGrafico.getWidth()/(esqID.getX()-esqSI.getX())
				,JPanelGrafico.getHeight()-( (pt.getY()-esqID.getY())*JPanelGrafico.getHeight()/(esqSI.getY()-esqID.getY()) )
				);
	}

	/**
	 * Convierte punto en el mundo real a punto en la pantalla.
	 * @param x coordenada X del punto
	 * @param y coordenada Y del punto
	 * @return punto en pantalla
	 */
	protected Point2D.Double point2Pixel(double x, double y) {
		return new Point2D.Double(
				( x - esqSI.getX())*JPanelGrafico.getWidth()/(esqID.getX()-esqSI.getX())
				,JPanelGrafico.getHeight()-( (y-esqID.getY())*JPanelGrafico.getHeight()/(esqSI.getY()-esqID.getY()) )
				);
	}

	protected Point2D.Double pixel2Point(Point2D px) {
		return new Point2D.Double(
				(double)px.getX() * (esqID.getX()-esqSI.getX()) / JPanelGrafico.getWidth() + esqSI.getX()
				,(double)(JPanelGrafico.getHeight()-px.getY()) * (esqSI.getY()-esqID.getY()) / JPanelGrafico.getHeight() + esqID.getY()
				);
	}
	

    protected Double pixel2Point(int x, int y) {
		return new Point2D.Double(
				(double)x * (esqID.getX()-esqSI.getX()) / JPanelGrafico.getWidth() + esqSI.getX()
				,(double)(JPanelGrafico.getHeight()-y) * (esqSI.getY()-esqID.getY()) / JPanelGrafico.getHeight() + esqID.getY()
				);
	}

	/**
	 * Crea parte grafica junto con slider de zoom
	 * @param distanciaMaxima Distancia máxima del gráfico
	 */
	public PanelBarrido(short distanciaMaxima) {
		distMax=80;
		if(distanciaMaxima>0 && distanciaMaxima<=80)
			distMax=distanciaMaxima;
		
		//setLayout(new BoxLayout(this, BoxLayout.LINE_AXIS));	
		
		setLayout(new BorderLayout(3,3));
		
		esqID=new Point2D.Double();
		esqSI=new Point2D.Double();
		restaurar=true;	//se actualizan las esquinas la primera vez
	
		//Primero el Panel
		JPanelGrafico=new JPanel() {

			protected void paintComponent(Graphics g0) {
				Graphics2D g=(Graphics2D)g0;
				super.paintComponent(g);
				if(restaurar) {
					//restauramos las esquinas
					double fact=(double)JPanelGrafico.getHeight()/(double)JSliderZoom.getValue();
					esqSI.setLocation(-JPanelGrafico.getWidth()/2/fact, JPanelGrafico.getHeight()/fact);
					esqID.setLocation(+JPanelGrafico.getWidth()/2/fact, 0);
                    distanciaVista=Math.sqrt(esqSI.getX()*esqSI.getX()+esqSI.getY()*esqSI.getY());
					restaurar=false;
				}

				//Cálculo del paso para regla y regillas
				//Permitimos paso de decimas de metros
				double pasoD=(double)JSliderZoom.getValue()/10.0; //paso en unidades reales
				int paso;  //distancia en decimas de metro
				if(pasoD>5) paso=100;
				else if (pasoD>2) paso=50;
				else if (pasoD>1) paso=20;
				else if (pasoD>0.5) paso=10;
				else if (pasoD>0.2) paso=5;
				else if (pasoD>0.1) paso=2;
				else paso=1;

				//puntos de referencia
				Point2D.Double pxCentro=point2Pixel(0.0,0.0); 
				Point2D.Double pxFrenteCentro=point2Pixel(0.0,(double)JSliderZoom.getValue());

				//Vemos la distancia máxima visible
				Point2D.Double ptoEsquina=pixel2Point(0, 0); //punto más alejado es esquina superior izda
				double distMaxVisible=Math.sqrt(ptoEsquina.getX()*ptoEsquina.getX()
						+ptoEsquina.getY()*ptoEsquina.getY());  //hipotenusa

				//Definición de las lineas de regla y rejillas
				g.setColor(Color.GRAY); //en color gris
				final float dash1[] = {10.0f};
				final BasicStroke dashed = new BasicStroke(0.5f, 
						BasicStroke.CAP_BUTT, 
						BasicStroke.JOIN_MITER, 
						10.0f, dash1, 0.0f);
				g.setStroke(dashed);

				if(jcbRegla.isSelected()) {
					g.draw(new Line2D.Double(pxCentro.getX(), pxCentro.getY()
							, pxFrenteCentro.getX(), pxFrenteCentro.getY()));
					//Ponemos etiquetas de distancias
					for(double ma=((double)paso)/10; ma<=distMaxVisible;ma+=((double)paso)/10) {
						Point2D.Double pxPtoMarca=point2Pixel(0.0,ma);
						g.draw(new Line2D.Double(pxPtoMarca.getX()-4.0, pxPtoMarca.getY()
								, pxPtoMarca.getX()+4, pxPtoMarca.getY()));
						String etiqueta=String.valueOf(ma);
						if (paso>=10) 
							//sin decimales
							etiqueta=etiqueta.substring(0,etiqueta.indexOf('.'));
						else
							//con 1 decimal
							etiqueta=etiqueta.substring(0,etiqueta.indexOf('.')+2);
						g.drawString(etiqueta,(float)pxPtoMarca.getX()+6,(float)pxPtoMarca.getY()+4);
					}
				}
				//Rejilla radial
				if(jcbRejillaRadial.isSelected()) {
					for(double ma=((double)paso)/10; ma<=distMaxVisible;ma+=((double)paso)/10) {
						Point2D.Double pxEsqSI=point2Pixel(-ma,ma)
						,pxEsqID=point2Pixel(+ma,-ma);
						g.draw(new Arc2D.Double(pxEsqSI.getX(),pxEsqSI.getY()
								,pxEsqID.getX()-pxEsqSI.getX(),pxEsqID.getY()-pxEsqSI.getY()
								,0.0, 180.0
								,Arc2D.OPEN));

					}

				}
				//Rejilla recta
				if(jcbRejillaRecta.isSelected()) {
					for(double ma=((double)paso)/10; ma<=distMaxVisible;ma+=((double)paso)/10) {
						Point2D.Double pxEsqSI=point2Pixel(-ma,ma)
						,pxEsqID=point2Pixel(+ma,0);
						g.draw(new Rectangle2D.Double(pxEsqSI.getX(),pxEsqSI.getY()
								,pxEsqID.getX()-pxEsqSI.getX(),pxEsqID.getY()-pxEsqSI.getY()
						));
					}
				}
				
				//cemicírculo central
				g.setStroke(new BasicStroke());
				g.setColor(Color.WHITE);
				g.draw(new Arc2D.Double(pxCentro.getX()-TamMarca/2, pxCentro.getY()-TamMarca/2 //esquina rectángulo
						, TamMarca, TamMarca //Tamaño rectángulo
						, 0, 180 //rango de ángulos
						,Arc2D.CHORD //ralla entre los extremos
				));
				cosasAPintar(g0);
			}
		};
		JPanelGrafico.setMinimumSize(new Dimension(400,400));
		JPanelGrafico.setSize(new Dimension(400,400));
		JPanelGrafico.setBackground(Color.BLACK);
		JPanelGrafico.addMouseListener(this);
		add(JPanelGrafico,BorderLayout.CENTER);
		
		//después (a la dercha) el slider
		JSliderZoom=new JSlider(SwingConstants.VERTICAL,0,distMax,distMax);
		JSliderZoom.setMajorTickSpacing(10);
		JSliderZoom.setMinorTickSpacing(5);
		JSliderZoom.setPaintLabels(true);
		JSliderZoom.setPaintTicks(true);
		JSliderZoom.addChangeListener(this);
		add(JSliderZoom,BorderLayout.LINE_END);
		
		{
			Dimension sepH=new Dimension(15,0);
			//Abajo los checks para mostrar o no las zonas y el barrido
			jpChecks=new JPanel();
			jpChecks.setLayout(new BoxLayout(jpChecks,BoxLayout.LINE_AXIS));
			jpChecks.setBorder(
					BorderFactory.createCompoundBorder(
							BorderFactory.createEmptyBorder(5, 5, 5, 5)
							,BorderFactory.createLineBorder(Color.BLACK)
					)
			);
			
			jpChecks.add(Box.createHorizontalStrut(15));
			
			jpChecks.add(new JLabel("Mostrar: "));
			
			jpChecks.add(Box.createHorizontalStrut(15));

			jcbRegla=new JCheckBox("Regla",true);
			jcbRegla.addChangeListener(this);
			jpChecks.add(jcbRegla);

			jcbRejillaRadial=new JCheckBox("Rejilla Radial",true);
			jcbRejillaRadial.addChangeListener(this);
			jpChecks.add(jcbRejillaRadial);

			jcbRejillaRecta=new JCheckBox("Rejilla Recta",false);  //recta no seleccionada
			jcbRejillaRecta.addChangeListener(this);
			jpChecks.add(jcbRejillaRecta);

			add(jpChecks,BorderLayout.PAGE_END);
		}
		
	}

	protected void cosasAPintar(Graphics g0) {
		// TODO Apéndice de método generado automáticamente
		
	}
	
	/**	
	 * Atiende los cambios en el JSlider y de los check box de barrido y zona.
	 * En todos los casos hay que repintar.
	 */
	public void stateChanged(ChangeEvent arg0) {
		//Impedimos llegue a 0 
		if(JSliderZoom.getValue()==0)
			JSliderZoom.setValue(1);
		// Mandamos repintar el panel del gráfico
		JPanelGrafico.repaint();
		restaurar=true;
	}
    /**
     * Doble click del boton 1 vuelve a la presentación normal
     */
	public void mouseClicked(MouseEvent even) {
		if(even.getButton()!=MouseEvent.BUTTON1 || even.getClickCount()!=2)
			return;
//		System.out.println("Clickeado Boton "+even.getButton()
//				+" en posición: ("+even.getX()+","+even.getY()+") "
//				+even.getClickCount()+" veces");
		restaurar=true;
		JPanelGrafico.repaint();
	}
    /**
     * Almacena donde se pulsa el botón 1 en {@link #evenPulsa}
     */
	public void mousePressed(MouseEvent even) {
		if(even.getButton()!=MouseEvent.BUTTON1)
			return;
//		Point2D.Double ptPulsa=pixel2Point(even.getX(),even.getY());
//		System.out.println("Pulsado Boton "+even.getButton()
//				+" en posición: ("+even.getX()+","+even.getY()+")"
//				+"  ("+ptPulsa.getX()+","+ptPulsa.getY()+")  "
//				+" distancia: "+ptPulsa.distance(new Point2D.Double(0,0))
//				);
		evenPulsa = even;
	}

	/**
     * Al soltar el botón 1 se mira si se a seleccionado rectángulo 
     * y se reescala a lo seleccionado
     */
	public void mouseReleased(MouseEvent even) {
		if(even.getButton()!=MouseEvent.BUTTON1)
			return;
//		System.out.println("Soltado Boton "+even.getButton()
//				+" en posición: ("+even.getX()+","+even.getY()+")");
		//Creamos rectángulo si está suficientemente lejos
		if(even.getX()-evenPulsa.getX()>50 
				&& even.getY()-evenPulsa.getY()>50) {
			//como se usan las esquinas actuales para calcular las nuevas sólo podemos modificarlas 
			// después
			Point2D.Double nuevaEsqSI=pixel2Point( new Point2D.Double(evenPulsa.getX(),evenPulsa.getY()) );
			esqID.setLocation(pixel2Point( new Point2D.Double(even.getX(),even.getY()) ));
			esqSI.setLocation(nuevaEsqSI);
			JPanelGrafico.repaint();
//			System.out.println("Puntos:  SI ("+ esqSI.getX()+ ","+esqSI.getY() +") "
//					+"  ID ("+esqID.getX()+","+esqID.getY()+")"
//					+"   ("+JPanelGrafico.getWidth()+","+JPanelGrafico.getHeight()+")"
//					);
		}
	}

	/** No hacenos nada */
	public void mouseEntered(MouseEvent arg0) {
		// TODO Apéndice de método generado automáticamente
		
	}

	/** no hacemos nada */
	public void mouseExited(MouseEvent arg0) {
		// TODO Apéndice de método generado automáticamente
		
	}


	public void repaint() {
		super.repaint();
		if (JPanelGrafico!=null)
			JPanelGrafico.repaint();
	}
}
