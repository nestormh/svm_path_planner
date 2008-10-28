package sibtra.util;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;


/**
 * Clase que presenta mapa con eje X vertical hacia arriba (Norte) y eje Y horizontal 
 * a la izda (Oeste)
 * @author alberto
 */
public class PanelMapa extends JPanel implements MouseListener, ActionListener {
	
	/** Tamaño (en pixeles) de los ejes a pintar en el panel */
	protected static final int TamEjes = 50;
	
	/** Panel de la gráfica */
	JPanel JPanelGrafico;
	
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

	/** Banderín para indicar que hay que recalcular las esquinas */
	private boolean restaurar;

	/** Evento cuando se pulsó el ratón, se necesita para hacer los cálculos al soltar y hacer el zoom */
	private MouseEvent evenPulsa;
	/** Evento cuando se pulsó el ratón con el SHIFT, establece la posición deseada */
	private MouseEvent evenPos;


	private JComboBox jcbEscalas;
	String[] escalasS={ "0.5 m","1 m","2 m", "5 m","10 m","20 m","50 m", "100 m", "500 m" };
	double[] escalasD={  0.5,    1,    2   ,  5   , 10   , 20   , 50   ,  100   ,  500    };  

	protected boolean escalado;

	private JPanel jpSur;

	protected boolean centrar;

	protected double[] centro;

	private JButton jbCentrar;


	/** Número de pixeles en pantalla que representa la escala seleccionada */
	protected static final int pixelEscala=100;
	
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
	protected Point2D.Double point2Pixel(double x, double y) {
		return new Point2D.Double(
				(esqSI.getY()-y)*JPanelGrafico.getWidth()/(esqSI.getY()-esqID.getY())
				,
				JPanelGrafico.getHeight()-((x-esqID.getX())*JPanelGrafico.getHeight()/(esqSI.getX()-esqID.getX()))
				);
	}

	/** @see #point2Pixel(double, double) */
	protected Point2D point2Pixel(double[] ds) {
		return point2Pixel(ds[0], ds[1]);
	}

	/** @see #point2Pixel(double, double) */
	protected Point2D.Double pixel2Point(Point2D px) {
		return pixel2Point(px.getX(), px.getY());
	}
	
	
	/**	
	 * Dado pixel de la pantalla determina a que punto del mundo real corresponde. 
	 * @param x coordenada x del pixel
	 * @param y coordenada y del pixel
	 * @return punto en el mundo real
	 */
	protected Point2D.Double pixel2Point(double x, double y) {
		return new Point2D.Double(
				(JPanelGrafico.getHeight()-y)*(esqSI.getX()-esqID.getX())/JPanelGrafico.getHeight() + esqID.getX()
				,
				esqSI.getY()-x*(esqSI.getY()-esqID.getY())/JPanelGrafico.getWidth()
				 );
    }


    /**
     * Constructor 
     * @param rupas Ruta pasada.
     */
	public PanelMapa() {
		// TODO Apéndice de constructor generado automáticamente
		setLayout(new BorderLayout(3,3));
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
				if(restaurar) {
					double f=escalasD[jcbEscalas.getSelectedIndex()]/pixelEscala;
					double lx=f*getHeight();
					double ly=f*getWidth();
					
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
				}
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
				cosasAPintar(g0);
			}
		};
		JPanelGrafico.setMinimumSize(new Dimension(400,400));
		JPanelGrafico.setSize(new Dimension(400,400));
		JPanelGrafico.setBackground(Color.BLACK);
		JPanelGrafico.addMouseListener(this);
		JPanelGrafico.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
		add(JPanelGrafico,BorderLayout.CENTER);
		
		{
			jpSur=new JPanel();
			
			jpSur.add(new JLabel("Escala"));
			jcbEscalas=new JComboBox(escalasS);
			jcbEscalas.setSelectedIndex(4);
			jcbEscalas.addActionListener(this);
			jpSur.add(jcbEscalas);
			
			jbCentrar=new JButton("Centrar");
			jbCentrar.addActionListener(this);
			jpSur.add(jbCentrar);
			
			
			add(jpSur,BorderLayout.SOUTH);
			
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
		System.out.println("Clickeado Boton "+even.getButton()
				+" en posición: ("+even.getX()+","+even.getY()+") "
				+even.getClickCount()+" veces");
		restaurar=true;
		JPanelGrafico.repaint();
	}
	
    /**
     * Almacena donde se pulsa el botón 1 en {@link #evenPulsa}
     */
	public void mousePressed(MouseEvent even) {
		evenPulsa=null;
		if(even.getButton()==MouseEvent.BUTTON1) {
			//no necesitamos distinguir si hubo o no SHFIT
			Point2D.Double ptPulsa=pixel2Point(even.getX(),even.getY());
			System.out.println("Pulsado Boton "+even.getButton()
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
				System.out.println("Soltado Boton "+even.getButton()
						+" en posición: ("+even.getX()+","+even.getY()+")");
				if(evenPulsa!=null) {
					System.out.println("Soltado Boton "+even.getButton()
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
						JPanelGrafico.repaint();
						System.out.println("Puntos:  SI ("+ esqSI.getX()+ ","+esqSI.getY() +") "
								+"  ID ("+esqID.getX()+","+esqID.getY()+")"
								+"   ("+JPanelGrafico.getWidth()+","+JPanelGrafico.getHeight()+")"
						);
					}
				}
			} else {
				//soltado boton 1 sin Shift, queremos desplazar el centro
				System.out.println("Soltado Boton "+even.getButton()
						+" SIN Shift en posición: ("+even.getX()+","+even.getY()+")");
				Point2D.Double ptoPulsado=pixel2Point( new Point2D.Double(evenPulsa.getX(),evenPulsa.getY()) );
				Point2D.Double ptoSoltado=pixel2Point( new Point2D.Double(even.getX(),even.getY()) );
				//desplazamos el centro en lo que se ha desplazado el cursor
				centro[0]+=-ptoSoltado.getX()+ptoPulsado.getX();
				centro[1]+=-ptoSoltado.getY()+ptoPulsado.getY();
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


	/** Restaura para que se vea todo y repinta el mapa */
	public void restaura() {
		restaurar=true;
		//programamos la actualizacion de la ventana
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();			
			}
		});
	}
	

	/** Restaura y repinta cuando se selecciona una nueva escala */
	public void actionPerformed(ActionEvent ae) {
		if(ae.getSource()==jcbEscalas) {
			//solo repintamos restaurando
			restaurar=true;
			JPanelGrafico.repaint();
		}
		if(ae.getSource()==jbCentrar) {
			//tenemos que centrar
			centrar=true; restaurar=true;
			JPanelGrafico.repaint();
		}
	}

	public void fijarCentro(double xc,double yc) {
		centro[0]=xc;
		centro[1]=yc;
		restaurar=true;
		JPanelGrafico.repaint();
	}

}
