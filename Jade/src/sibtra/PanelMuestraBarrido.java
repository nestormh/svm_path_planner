/*
 * Creado el 22/02/2008
 *
 * Creado por Alberto Hamilcon con Eclipse
 */
package sibtra;

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

public class PanelMuestraBarrido extends JPanel implements ChangeListener, MouseListener {
	
	protected static final int TamMarca = 40;

	/**
	 * Para variar zoom
	 */
	private JSlider JSliderZoom;
	
	/**
	 * Panel donde pintar
	 */
	private JPanel	JPanelGrafico;
	
	private JCheckBox jcbBarrido;
	private JCheckBox jcbZonaA;
	private JCheckBox jcbZonaB;
	private JCheckBox jcbZonaC;
	
	private JCheckBox jcbRejillaRadial;
	private JCheckBox jcbRejillaRecta;

	/**
	 * Distancia maxima a representar
	 */
	private short distMax;
	
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

	/**
	 * Convierte punto en el mundo real a punto en la pantalla.
	 * @param pt punto del mundo real
	 * @return punto en pantalla
	 */
	private Point2D.Double point2Pixel(Point2D pt) {
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
	private Point2D.Double point2Pixel(double x, double y) {
		return new Point2D.Double(
				( x - esqSI.getX())*JPanelGrafico.getWidth()/(esqID.getX()-esqSI.getX())
				,JPanelGrafico.getHeight()-( (y-esqID.getY())*JPanelGrafico.getHeight()/(esqSI.getY()-esqID.getY()) )
				);
	}

	private Point2D.Double pixel2Point(Point2D px) {
		return new Point2D.Double(
				(double)px.getX() * (esqID.getX()-esqSI.getX()) / JPanelGrafico.getWidth() + esqSI.getX()
				,(double)(JPanelGrafico.getHeight()-px.getY()) * (esqSI.getY()-esqID.getY()) / JPanelGrafico.getHeight() + esqID.getY()
				);
	}
	

    private Double pixel2Point(int x, int y) {
		return new Point2D.Double(
				(double)x * (esqID.getX()-esqSI.getX()) / JPanelGrafico.getWidth() + esqSI.getX()
				,(double)(JPanelGrafico.getHeight()-y) * (esqSI.getY()-esqID.getY()) / JPanelGrafico.getHeight() + esqID.getY()
				);
	}

	/**
	 * Crea parte grafica junto con slider de zoom
	 * @param distanciaMaxima Distancia máxima del gráfico
	 */
	public PanelMuestraBarrido(short distanciaMaxima) {
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
					restaurar=false;
				}

				g.setColor(Color.WHITE);
				Point2D.Double pxCentro=point2Pixel(0.0,0.0); 
				g.draw(new Arc2D.Double(pxCentro.getX()-TamMarca/2, pxCentro.getY()-TamMarca/2 //esquina rectángulo
						, TamMarca, TamMarca //Tamaño rectángulo
						, 0, 180 //rango de ángulos
						,Arc2D.CHORD //ralla entre los extremos
						));
				
				{   //etiquetas de distancia
					//Cosas comunes para cualquier rejilla
					g.setColor(Color.GRAY); //en color gris
					Point2D.Double pxFrenteCentro=point2Pixel(0.0,(double)JSliderZoom.getValue());
					g.draw(new Line2D.Double(pxCentro.getX(), pxCentro.getY()
							, pxFrenteCentro.getX(), pxFrenteCentro.getY()));
					final float dash1[] = {10.0f};
				    final BasicStroke dashed = new BasicStroke(0.5f, 
				                                          BasicStroke.CAP_BUTT, 
				                                          BasicStroke.JOIN_MITER, 
				                                          10.0f, dash1, 0.0f);
				    g.setStroke(dashed);

				    //circulos de distancia, que se vean al menos 10
				    Point2D.Double ptoEsquina=pixel2Point(0, 0); //punto más alejado es esquina superior izda
				    double distMax=Math.sqrt(ptoEsquina.getX()*ptoEsquina.getX()
				    		+ptoEsquina.getY()*ptoEsquina.getY());  //hipotenusa
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
				    //Ponemos etiquetas de distancias
					for(double ma=((double)paso)/10; ma<=distMax;ma+=((double)paso)/10) {
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
					//Rejilla radial
					if(jcbRejillaRadial.isSelected()) {
						for(double ma=((double)paso)/10; ma<=distMax;ma+=((double)paso)/10) {
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
						for(double ma=((double)paso)/10; ma<=distMax;ma+=((double)paso)/10) {
							Point2D.Double pxEsqSI=point2Pixel(-ma,ma)
								,pxEsqID=point2Pixel(+ma,0);
						    g.draw(new Rectangle2D.Double(pxEsqSI.getX(),pxEsqSI.getY()
						    			,pxEsqID.getX()-pxEsqSI.getX(),pxEsqID.getY()-pxEsqSI.getY()
						    			));
						}
					}
				}
				
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
			JPanel jpC=new JPanel();
			jpC.setLayout(new BoxLayout(jpC,BoxLayout.LINE_AXIS));
			jpC.setBorder(
					BorderFactory.createCompoundBorder(
							BorderFactory.createEmptyBorder(5, 5, 5, 5)
							,BorderFactory.createLineBorder(Color.BLACK)
					)
			);
			
			jpC.add(Box.createHorizontalStrut(15));
			
			jpC.add(new JLabel("Mostrar: "));
			
			jpC.add(Box.createHorizontalStrut(15));

			jcbBarrido=new JCheckBox("Barrido",true);
			jcbBarrido.addChangeListener(this);
			jpC.add(jcbBarrido);
			
			jpC.add(Box.createHorizontalStrut(15));
			
			jcbZonaA=new JCheckBox("Zona A",true);
			jcbZonaA.addChangeListener(this);
			jpC.add(jcbZonaA);
			
			jpC.add(Box.createHorizontalStrut(15));
			
			jcbZonaB=new JCheckBox("Zona B",true);
			jcbZonaB.addChangeListener(this);
			jpC.add(jcbZonaB);
			
			jpC.add(Box.createHorizontalStrut(15));
			
			jcbZonaC=new JCheckBox("Zona C",true);
			jcbZonaC.addChangeListener(this);
			jpC.add(jcbZonaC);
			
			jpC.add(Box.createHorizontalStrut(15));
			
			jcbRejillaRadial=new JCheckBox("Rejilla Radial",true);
			jcbRejillaRadial.addChangeListener(this);
			jpC.add(jcbRejillaRadial);

			jcbRejillaRecta=new JCheckBox("Rejilla Recta",false);  //recta no seleccionada
			jcbRejillaRecta.addChangeListener(this);
			jpC.add(jcbRejillaRecta);

			add(jpC,BorderLayout.PAGE_END);
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
		
		if(barr==null)
			return;
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
		if(zona.isConjunto1()){
			//es del conjunto 1
			if(zona.getQueZona()==ZonaLMS.ZONA_A) Zona1A=zona;
			if(zona.getQueZona()==ZonaLMS.ZONA_B) Zona1B=zona;
			if(zona.getQueZona()==ZonaLMS.ZONA_C) Zona1C=zona;
		} else {
			if(zona.getQueZona()==ZonaLMS.ZONA_A) Zona2A=zona;
			if(zona.getQueZona()==ZonaLMS.ZONA_B) Zona2B=zona;
			if(zona.getQueZona()==ZonaLMS.ZONA_C) Zona2C=zona;
		}
		JPanelGrafico.repaint();
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
		System.out.println("Clickeado Boton "+even.getButton()
				+" en posición: ("+even.getX()+","+even.getY()+") "
				+even.getClickCount()+" veces");
		restaurar=true;
		JPanelGrafico.repaint();
	}
    /**
     * Sólo nos interesan pulsaciones del boton 1
     */
	public void mousePressed(MouseEvent even) {
		if(even.getButton()!=MouseEvent.BUTTON1)
			return;
		Point2D.Double ptPulsa=pixel2Point(even.getX(),even.getY());
		System.out.println("Pulsado Boton "+even.getButton()
				+" en posición: ("+even.getX()+","+even.getY()+")"
				+"  ("+ptPulsa.getX()+","+ptPulsa.getY()+")  "
				+" distancia: "+ptPulsa.distance(new Point2D.Double(0,0))
				);
		evenPulsa = even;
	}

	/**
     * Sólo nos interesan al soltar el boton 1
     */
	public void mouseReleased(MouseEvent even) {
		if(even.getButton()!=MouseEvent.BUTTON1)
			return;
		System.out.println("Soltado Boton "+even.getButton()
				+" en posición: ("+even.getX()+","+even.getY()+")");
		//Creamos rectángulo si está suficientemente lejos
		if(even.getX()-evenPulsa.getX()>50 
				&& even.getY()-evenPulsa.getY()>50) {
			//como se usan las esquinas actuales para calcular las nuevas sólo podemos modificarlas 
			// después
			Point2D.Double nuevaEsqSI=pixel2Point( new Point2D.Double(evenPulsa.getX(),evenPulsa.getY()) );
			esqID.setLocation(pixel2Point( new Point2D.Double(even.getX(),even.getY()) ));
			esqSI.setLocation(nuevaEsqSI);
			JPanelGrafico.repaint();
			System.out.println("Puntos:  SI ("+ esqSI.getX()+ ","+esqSI.getY() +") "
					+"  ID ("+esqID.getX()+","+esqID.getY()+")"
					+"   ("+JPanelGrafico.getWidth()+","+JPanelGrafico.getHeight()+")"
					);
		}
	}

	public void mouseEntered(MouseEvent arg0) {
		// TODO Apéndice de método generado automáticamente
		
	}

	public void mouseExited(MouseEvent arg0) {
		// TODO Apéndice de método generado automáticamente
		
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
		
		PMB.repaint();
		
	}


}
