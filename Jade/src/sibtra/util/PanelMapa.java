package sibtra.util;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;


/**
 * Clase que presenta mapa con eje X vertical hacia arriba (Norte) y eje Y horizontal 
 * a la izda (Oeste)
 * @author alberto
 */
@SuppressWarnings("serial")
public class PanelMapa extends JPanel implements MouseListener, ActionListener {
	
	/** Tamaño (en pixeles) de los ejes a pintar en el panel */
	protected static final int TamEjes = 50;
	
	/** Panel de la gráfica */
	JPanel JPanelGrafico;
	
	/**
	 * Coordenadas de la esquina superior izquierda.
	 * En unidades mundo real.
	 */
	protected Point2D esqSI;
	
	/**
	 * Coordenadas de la esquina superior izquierda.
	 * En unidades mundo real.
	 */
	protected Point2D esqID;

	/** Banderín para indicar que hay que recalcular las esquinas */
	private boolean restaurar;

	/** Evento cuando se pulsó el ratón, se necesita para hacer los cálculos al soltar y hacer el zoom */
	private MouseEvent evenPulsa;

	private JComboBox jcbEscalas;
	String[] escalasS={ "0.5 m","1 m","2 m", "5 m","10 m","20 m","50 m", "100 m", "500 m" };
	double[] escalasD={  0.5,    1,    2   ,  5   , 10   , 20   , 50   ,  100   ,  500    };  

	protected boolean escalado;

	protected JPanel jpSur;

	protected boolean centrar;

	protected double[] centro;

	private JButton jbCentrar;

	private JCheckBox jcbRejilla;


	/** Número de pixeles en pantalla que representa la escala seleccionada */
	protected static final int pixelEscala=100;
	
	/**
	 * @return the jPanelGrafico
	 */
	public JPanel getJPanelGrafico() {
		return JPanelGrafico;
	}

	/** @return el valor de la escala solicitiada o NaN si no hay escala valida */
	public double getEscala() {
		if(jcbEscalas.isEnabled())
			return escalasD[jcbEscalas.getSelectedIndex()];
		else
			return Double.NaN;
	}
	/**
	 * Convierte punto en el mundo real a punto en la pantalla. 
	 * EN VERTICAL EL EJE X hacia la izquierda el eje Y. 
	 * @param pt punto del mundo real
	 * @return punto en pantalla
	 */
	protected Point2D.Double point2Pixel(Point2D pt) {
		return point2Pixel(pt.getX(), pt.getY()); 
	}

	
	/**
	 * Convierte punto en el mundo real a punto en la pantalla.
	 *  EJE X vertical hacia arriba, Eje Y horizontal hacia la izquierda
	 * @param x coordenada X del punto
	 * @param y coordenada Y del punto
	 * @return punto en pantalla
	 */
	public Point2D.Double point2Pixel(double x, double y) {
		return new Point2D.Double(
				(esqSI.getY()-y)*JPanelGrafico.getWidth()/(esqSI.getY()-esqID.getY())
				,
				JPanelGrafico.getHeight()-((x-esqID.getX())*JPanelGrafico.getHeight()/(esqSI.getX()-esqID.getX()))
				);
	}

	/** @see #point2Pixel(double, double) */
	public Point2D point2Pixel(double[] ds) {
		return point2Pixel(ds[0], ds[1]);
	}

	/** @see #point2Pixel(double, double) */
	public Point2D.Double pixel2Point(Point2D px) {
		return pixel2Point(px.getX(), px.getY());
	}
	
	/**	
	 * Dado pixel de la pantalla determina a que punto del mundo real corresponde. 
	 * @param x coordenada x del pixel
	 * @param y coordenada y del pixel
	 * @return punto en el mundo real
	 */
	public Point2D.Double pixel2Point(double x, double y) {
		return new Point2D.Double(
				(JPanelGrafico.getHeight()-y)*(esqSI.getX()-esqID.getX())/JPanelGrafico.getHeight() + esqID.getX()
				,
				esqSI.getY()-x*(esqSI.getY()-esqID.getY())/JPanelGrafico.getWidth()
				 );
    }


    /**
     * Constructor 
     */
	public PanelMapa() {
		esqID=new Point2D.Double();
		esqSI=new Point2D.Double();
		centro=new double[2];
		restaurar=true;	//se actualizan las esquinas la primera vez
		centrar=true;

		//Primero el Panel
		JPanelGrafico=new JPanel() {
			protected void paintComponent(Graphics g0) {
				Graphics2D g=(Graphics2D)g0;
				super.paintComponent(g);
				if(centrar) {					
					double axis[]=limites();
					//restauramos las esquinas
					double minCX;
					double maxCX;
					double minCY;
					double maxCY;
					if(axis!=null && axis.length>=4 && axis[0]<axis[1] && axis[2]<axis[3]) {
					minCX=axis[0];
					maxCX=axis[1];
					minCY=axis[2];
					maxCY=axis[3];
					} else {
						System.err.println("Ejes devueltos por la función límite no son correctos");
						minCX=-10;	maxCX=10;	minCY=-10;	maxCY=10;
					}
					//punto medio del rango de X y de Y
					centro[0]=(minCX+maxCX)/2;
					centro[1]=(minCY+maxCY)/2;
					centrar=false;
				}
				double escala=escalasD[jcbEscalas.getSelectedIndex()];
				if(restaurar) {
					double lx=escala/pixelEscala*getHeight();
					double ly=escala/pixelEscala*getWidth();
					
					esqSI.setLocation(centro[0]+lx/2,centro[1]+ly/2);
					esqID.setLocation(centro[0]-lx/2,centro[1]-ly/2);

					restaurar=false;
					escalado=true;
				}
				if (escalado) {
					//pintamos referencia de escala
					g.setColor(Color.WHITE);
					float pxSep=20;
					float altE=4;
					float xCentE=getWidth()-pixelEscala/2-pxSep;
					float yCentE=getHeight()-pxSep;
					GeneralPath gpE=new GeneralPath(GeneralPath.WIND_EVEN_ODD,4);
					gpE.moveTo(xCentE-pixelEscala/2, yCentE-altE);
					gpE.lineTo(xCentE-pixelEscala/2, yCentE);
					gpE.lineTo(xCentE+pixelEscala/2, yCentE);
					gpE.lineTo(xCentE+pixelEscala/2, yCentE-altE);
					g.draw(gpE);
					g.drawString((String)jcbEscalas.getSelectedItem(), xCentE, yCentE-altE);
					jcbEscalas.setEnabled(true);
				} else
					jcbEscalas.setEnabled(false);
				{//Pintamos  ejes en 0,0
					g.setColor(Color.WHITE);
					Point2D.Double pxCentro=point2Pixel(0.0,0.0);
					GeneralPath ejes=new GeneralPath(GeneralPath.WIND_EVEN_ODD,3);
					ejes.moveTo((float)pxCentro.getX()-TamEjes, (float)pxCentro.getY());
					ejes.lineTo((float)pxCentro.getX(), (float)pxCentro.getY());
					ejes.lineTo((float)pxCentro.getX(), (float)pxCentro.getY()-TamEjes);
					g.draw(ejes);
					g.drawString("N",(float)pxCentro.getX()+3,(float)pxCentro.getY()-TamEjes+12);
					g.drawString("W",(float)pxCentro.getX()-TamEjes,(float)pxCentro.getY()-3);
				}
				
				//Rejilla recta
				if(jcbRejilla.isSelected()) {
					//Definición de las lineas de rejilla
					g.setColor(Color.GRAY); //en color gris
					final float dash1[] = {10.0f};
					final BasicStroke dashed = new BasicStroke(0.5f, 
							BasicStroke.CAP_BUTT, 
							BasicStroke.JOIN_MITER, 
							10.0f, dash1, 0.0f);
					g.setStroke(dashed);
					double xmin=Math.floor(esqID.getX()/escala)*escala;
					double xmax=Math.ceil(esqSI.getX()/escala)*escala;
					double ymin=Math.floor(esqID.getY()/escala)*escala;
					double ymax=Math.ceil(esqSI.getY()/escala)*escala;
					//verticales
					for(double ya=ymin; 
					ya<=ymax;
					ya+=escala) {
						Point2D.Double ptInf=point2Pixel(xmin,ya);
						Point2D.Double ptSup=point2Pixel(xmax,ya);
						g.draw(new Line2D.Double(ptInf,ptSup));
						g.drawString(String.format("%+1.0f",ya),(float)ptInf.getX()+6,getHeight()-4);
					}
					//horizontales
					for(double xa=xmin; 
					xa<=xmax;
					xa+=escala) {
						Point2D.Double ptIzdo=point2Pixel(xa, ymin);
						Point2D.Double ptDecho=point2Pixel(xa, ymax);
						g.draw(new Line2D.Double(ptIzdo,ptDecho));
						g.drawString(String.format("%+1.0f", xa),4,(float)ptDecho.getY()-6);
					}
				}

				cosasAPintar(g0);
			}
		};
		setLayout(new BoxLayout(this,BoxLayout.PAGE_AXIS));
		JPanelGrafico.setMinimumSize(new Dimension(50,50));
		//JPanelGrafico.setSize(new Dimension(400,400));
		//para que se expanda todo lo que pueda en el boxLayout
		JPanelGrafico.setPreferredSize(new Dimension(Short.MAX_VALUE,Short.MAX_VALUE));
//		JPanelGrafico.setMaximumSize(new Dimension(Short.MAX_VALUE,Short.MAX_VALUE));
		JPanelGrafico.setBackground(Color.BLACK);
		JPanelGrafico.addMouseListener(this);
		JPanelGrafico.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
		
		add(JPanelGrafico);
		
		{
			jpSur=new PanelFlow();
			jpSur.setLayout(new FlowLayout(FlowLayout.LEFT,3,3));
			
			jpSur.add(new JLabel("Escala"));
			jcbEscalas=new JComboBox(escalasS);
			jcbEscalas.setSelectedIndex(4);
			jcbEscalas.addActionListener(this);
			jpSur.add(jcbEscalas);
			
			jbCentrar=new JButton("Centrar");
			jbCentrar.addActionListener(this);
			jpSur.add(jbCentrar);
			
			jcbRejilla= new JCheckBox("Rejilla");
			jcbRejilla.addActionListener(this);  //solo para que se repinte si cambia
			jpSur.add(jcbRejilla);
			
			//Para depurar los tamaños
//			jpSur.setBorder(BorderFactory.createCompoundBorder(
//	                   BorderFactory.createLineBorder(Color.red),
//	                   jpSur.getBorder()));

			add(jpSur);
		}
		
	}

	/**
	 *  metodo en que se indican los límites de lo que este objeto tiene que pintar en el panel
	 *  @return array de 4 elementos (minX,maxX, minY,maxY) 
	 */
	protected double[] limites() {
		double axis[]={-10,12,-10,12};
		return axis;
	}


	/** Método donde los hijos añadirán las cosas que quieran pintar.
	 * Se invoca desde el paint() del panel gráfico.
	 */
    protected void cosasAPintar(Graphics g0) {
		
	}


	/**
     * Doble click del boton 1 vuelve a la presentación normal
     */
	public void mouseClicked(MouseEvent even) {
		if(even.getButton()!=MouseEvent.BUTTON1 || even.getClickCount()!=2)
			return;
		System.out.println(getClass().getName()+": Clickeado Boton "+even.getButton()
				+" en posición: ("+even.getX()+","+even.getY()+") "
				+even.getClickCount()+" veces");
		restaurar=true;
		JPanelGrafico.repaint();
	}
	
    /**
     * Almacena donde se pulsa el botón 1 en {@link #evenPulsa} si hay Shift o Control
     */
	public void mousePressed(MouseEvent even) {
		evenPulsa=null;
		if(even.getButton()==MouseEvent.BUTTON1 
				&& ( ((even.getModifiersEx()&MouseEvent.SHIFT_DOWN_MASK)!=0)
						|| ((even.getModifiersEx()&MouseEvent.CTRL_DOWN_MASK)!=0) 
				)
		) {
			//no necesitamos distinguir si hubo o no SHFIT
			Point2D.Double ptPulsa=pixel2Point(even.getX(),even.getY());
			System.out.println(getClass().getName()+": Pulsado Boton "+even.getButton()
					+" en posición: ("+even.getX()+","+even.getY()+")"
					+"  ("+ptPulsa.getX()+","+ptPulsa.getY()+")  "
					+" distancia: "+ptPulsa.distance(new Point2D.Double(0,0))
			);
			evenPulsa = even;
			//ponemos el cursor según tipo de pulsación
			if((even.getModifiersEx()&MouseEvent.SHIFT_DOWN_MASK)!=0)
				JPanelGrafico.setCursor(Cursor.getPredefinedCursor(Cursor.SE_RESIZE_CURSOR));
			else
				JPanelGrafico.setCursor(Cursor.getPredefinedCursor(Cursor.MOVE_CURSOR));
			return;
		}
	}

	/**
     * Al soltar el botón 1 se mira si se a seleccionado rectángulo 
     * y se reescala a lo seleccionado.
     * Termina el trabajo empezado en {@link #mousePressed(MouseEvent)}
     */
	public void mouseReleased(MouseEvent even) {
		JPanelGrafico.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
		if(even.getButton()==MouseEvent.BUTTON1)
			if ((even.getModifiersEx()&MouseEvent.SHIFT_DOWN_MASK)!=0) {
				//Si hay shift ponemos esquinas en recuadro marcado
				System.out.println(getClass().getName()+": Soltado Boton "+even.getButton()
						+" en posición: ("+even.getX()+","+even.getY()+")");
				if(evenPulsa!=null) {
					System.out.println(getClass().getName()+" Soltado Boton "+even.getButton()
							+" con Shift en posición: ("+even.getX()+","+even.getY()+")");
					//Creamos rectángulo si está suficientemente lejos
					if(even.getX()-evenPulsa.getX()>50 
							&& even.getY()-evenPulsa.getY()>50) {
						//como se usan las esquinas actuales para calcular las nuevas sólo podemos modificarlas 
						// después
						Point2D.Double nuevaEsqSI=pixel2Point( new Point2D.Double(evenPulsa.getX(),evenPulsa.getY()) );
						esqID.setLocation(pixel2Point( new Point2D.Double(even.getX(),even.getY()) ));
						esqSI.setLocation(nuevaEsqSI);
						escalado=false;
//						jcbEscalas.setEnabled(false);
						JPanelGrafico.repaint();
						System.out.println(getClass().getName()+" Puntos:  SI ("+ esqSI.getX()+ ","+esqSI.getY() +") "
								+"  ID ("+esqID.getX()+","+esqID.getY()+")"
								+"   ("+JPanelGrafico.getWidth()+","+JPanelGrafico.getHeight()+")"
						);
					}
				}
			} else if ((even.getModifiersEx()&MouseEvent.CTRL_DOWN_MASK)!=0) {
				//soltado boton 1 sin Shift, queremos desplazar el centro
				System.out.println(getClass().getName()+" Soltado Boton "+even.getButton()
						+" SIN Shift en posición: ("+even.getX()+","+even.getY()+")");
				if(evenPulsa!=null) {
					Point2D.Double ptoPulsado=pixel2Point( new Point2D.Double(evenPulsa.getX(),evenPulsa.getY()) );
					Point2D.Double ptoSoltado=pixel2Point( new Point2D.Double(even.getX(),even.getY()) );
					//desplazamos el centro en lo que se ha desplazado el cursor
					centro[0]+=-ptoSoltado.getX()+ptoPulsado.getX();
					centro[1]+=-ptoSoltado.getY()+ptoPulsado.getY();
				} else {
					System.out.println(getClass().getName()+" Se pulsó con algún modificador, evenPulsa==null ");
				}
				restaurar=true;
				JPanelGrafico.repaint();
			}
	}

	/** NO hacemos nada */
	public void mouseEntered(MouseEvent arg0) {
		// No hacemos nada por ahora
		
	}

	/** NO hacemos nada */
	public void mouseExited(MouseEvent arg0) {
		// No hacemos nada por ahora
		
	}

	

	/** Restaura y repinta cuando se selecciona una nueva escala */
	public void actionPerformed(ActionEvent ae) {
		if(ae.getSource()==jcbEscalas) {
			//solo repintamos restaurando
			restaurar=true;
		}
		if(ae.getSource()==jbCentrar) {
			//tenemos que centrar
			centrar=true; restaurar=true;
		}
		JPanelGrafico.repaint();
	}

	/** Fija el centro del mapa (en coordenadas del mundo real).
	 *  NO ACTUALIZA PRESENTACIÓN  (usar {@link #actualiza()})
	 */
	public void setCentro(double xc,double yc) {
		centro[0]=xc;
		centro[1]=yc;
		restaurar=true;
	}


	/** Restaura para que se vea todo y repinta el mapa */
	public void restaura() {
		restaurar=true;
		actualiza();
	}


	/** Centra el mapa con los límites. NO ACTUALIZA PRESENTACIÓN (usar {@link #actualiza()}) */
	public void centra() {
		restaurar=true;
		actualiza();
	}

	/** programa el repintado del panel */
	public void actualiza() {
		//TODO debería ser un repinta, dejar actualiza para la llamada interna?
		//programamos la actualizacion de la ventana
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();						
			}
		});
	}
	
}
