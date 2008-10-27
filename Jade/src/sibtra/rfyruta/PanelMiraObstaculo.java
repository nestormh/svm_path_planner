package sibtra.rfyruta;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.BorderFactory;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;

import sibtra.gps.Ruta;
import sibtra.lms.BarridoAngular;

/**
 * Clase amiga de {@link MiraObstaculo} que muestar graficamente resultado de sus cálculos.
 * Ponemos eje X vertical hacia arriba (Norte) y eje Y horizontal a la izda (Oeste)
 * @author alberto
 */
public class PanelMiraObstaculo extends JPanel implements MouseListener {
	
	/** Tamaño (en pixeles) de los ejes a pintar en el panel */
	protected static final int TamEjes = 50;

	/** Objeto {@link MiraObstaculo} del cual se obtiene toda la información a representar*/
	private MiraObstaculo MI;
	
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


	/** Si ya hay datos que representar (posición y barrido) */
	private boolean hayDatos;

	/** Largo del coche en metros */
	protected double largoCoche=2;
	/** ancho del coche en metros */
	protected double anchoCoche = 1;

	/** etiqueta que muestar la distancia al obstaculo */
	private JLabel jlDistLin;
	/** etiqueta que muestar la distancia al obstaculo */
	private JLabel jlDistCam;

	/** Etiqueta para indicar cuando estamos fuera del camino */
	private JLabel jlFuera;

	/**
	 * Convierte punto en el mundo real a punto en la pantalla. 
	 * EN VERTICAL EL EJE X hacia la izquierda el eje Y. 
	 * @param pt punto del mundo real
	 * @return punto en pantalla
	 */
	private Point2D.Double point2Pixel(Point2D pt) {
		return point2Pixel(pt.getX(), pt.getY()); 
	}

	
	/**
	 * Convierte punto en el mundo real a punto en la pantalla.
	 *  EJE X vertical hacia arriba, Eje Y horizontal hacia la izquierda
	 * @param x coordenada X del punto
	 * @param y coordenada Y del punto
	 * @return punto en pantalla
	 */
	private Point2D.Double point2Pixel(double x, double y) {
		return new Point2D.Double(
				(esqSI.getY()-y)*JPanelGrafico.getWidth()/(esqSI.getY()-esqID.getY())
				,
				JPanelGrafico.getHeight()-((x-esqID.getX())*JPanelGrafico.getHeight()/(esqSI.getX()-esqID.getX()))
				);
	}

	/** @see #point2Pixel(double, double) */
	private Point2D point2Pixel(double[] ds) {
		return point2Pixel(ds[0], ds[1]);
	}

	/** @see #point2Pixel(double, double) */
	private Point2D.Double pixel2Point(Point2D px) {
		return pixel2Point(px.getX(), px.getY());
	}
	
	
	/**	
	 * Dado pixel de la pantalla determina a que punto del mundo real corresponde. 
	 * @param x coordenada x del pixel
	 * @param y coordenada y del pixel
	 * @return punto en el mundo real
	 */
    private Point2D.Double pixel2Point(double x, double y) {
		return new Point2D.Double(
				(JPanelGrafico.getHeight()-y)*(esqSI.getX()-esqID.getX())/JPanelGrafico.getHeight() + esqID.getX()
				,
				esqSI.getY()-x*(esqSI.getY()-esqID.getY())/JPanelGrafico.getWidth()
				 );
    }


    /**
     * Constructor 
     * @param miObs Obejto {@link MiraObstaculo} del cual se obtendrá toda la información.
     */
	public PanelMiraObstaculo(MiraObstaculo miObs) {
		this.MI=miObs;
		setLayout(new BorderLayout(3,3));
		hayDatos=false;
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
					double minCX=min(0,MI.Tr,MI.Bd,MI.Bi);
//					minCX=Math.min(minCX, 0);
					double minCY=min(1,MI.Tr,MI.Bd,MI.Bi);
//					minCY=Math.min(minCY, 0);
					double maxCX=max(0,MI.Tr,MI.Bd,MI.Bi);
//					maxCX=Math.max(maxCX, 0);
					double maxCY=max(1,MI.Tr,MI.Bd,MI.Bi);
//					maxCY=Math.max(maxCY, 0);
					//Usamos mismo factor en ambas direcciones y centramos.
					double Dx=(maxCX-minCX);
					double Dy=(maxCY-minCY);
					double fx=Dx/getHeight();
					double fy=Dy/getWidth();
					double f=(fx>fy)?fx:fy;
					double lx=f*getHeight();
					double ly=f*getWidth();
					esqSI.setLocation(minCX+Dx+(lx-Dx)/2,minCY+Dy+(ly-Dy)/2);
					esqID.setLocation(minCX-(lx-Dx)/2,minCY-(ly-Dy)/2);

					restaurar=false;
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
				{
					//pintamos el trayecto
					g.setStroke(new BasicStroke());
					g.setColor(Color.YELLOW);
					g.draw(pathArrayXY(MI.Tr));
					//pintamos el borde derecho
					g.setColor(Color.BLUE);
					g.draw(pathArrayXY(MI.Bd));
					//pintamos el borde izquierdo
					g.setColor(Color.RED);
					g.draw(pathArrayXY(MI.Bi));
				}
				if(hayDatos) {
					//pixeles del pto actual se usa para trazar varias líneas
					Point2D pxPtoActual=point2Pixel(MI.posActual);
					
					{ //pintamos el barrido
						g.setStroke(new BasicStroke());
						g.setColor(Color.WHITE);
						//pintamos rango de puntos en camino
						GeneralPath perimetro = 
							new GeneralPath(GeneralPath.WIND_EVEN_ODD, MI.barr.numDatos());

						Point2D.Double px=point2Pixel(ptoRF2Point(0));
						perimetro.moveTo((float)px.getX(),(float)px.getY());
						for(int i=1; i<MI.barr.numDatos(); i++ ) {
							px=point2Pixel(ptoRF2Point(i));
							perimetro.lineTo((float)px.getX(),(float)px.getY());
						}
						g.draw(perimetro);
					
					}

					{//Posición y orientación del coche
						g.setStroke(new BasicStroke(3));
						g.setPaint(Color.GRAY);
						g.setColor(Color.GRAY);
						double[] esqDD={MI.posActual[0]+anchoCoche/2*Math.sin(MI.Yaw)
								,MI.posActual[1]-anchoCoche/2*Math.cos(MI.Yaw) };
						double[] esqDI={MI.posActual[0]-anchoCoche/2*Math.sin(MI.Yaw)
								,MI.posActual[1]+anchoCoche/2*Math.cos(MI.Yaw) };
						double[] esqPD={esqDD[0]-largoCoche*Math.cos(MI.Yaw)
								,esqDD[1]-largoCoche*Math.sin(MI.Yaw) };
						double[] esqPI={esqDI[0]-largoCoche*Math.cos(MI.Yaw)
								,esqDI[1]-largoCoche*Math.sin(MI.Yaw) };
						Point2D pxDD=point2Pixel(esqDD);
						Point2D pxDI=point2Pixel(esqDI);
						Point2D pxPD=point2Pixel(esqPD);
						Point2D pxPI=point2Pixel(esqPI);
						GeneralPath coche=new GeneralPath();
						coche.moveTo((float)pxDD.getX(),(float)pxDD.getY());
						coche.lineTo((float)pxPD.getX(),(float)pxPD.getY());
						coche.lineTo((float)pxPI.getX(),(float)pxPI.getY());
						coche.lineTo((float)pxDI.getX(),(float)pxDI.getY());
						coche.closePath();
						g.fill(coche);
						g.draw(coche);
					}
					//vemos si hay información de colisión
					if(!Double.isNaN(MI.dist)) {
						//estamos dentro del camino
						//Lineas indicando de donde empezó el barrido
						g.setStroke(new BasicStroke());
						g.setColor(Color.GRAY);
						g.draw(new Line2D.Double(pxPtoActual
								,point2Pixel(MI.Bd[MI.iptoDini])));
						g.draw(new Line2D.Double(pxPtoActual
								,point2Pixel(MI.Bd[MI.iptoD])));
						g.draw(new Line2D.Double(pxPtoActual
								,point2Pixel(MI.Bi[MI.iptoIini])));
						g.draw(new Line2D.Double(pxPtoActual
								,point2Pixel(MI.Bi[MI.iptoI])));


						g.setStroke(new BasicStroke(2));
						g.setColor(Color.WHITE);
						//los de la derecha e izquierda que están libres
						g.draw(pathArrayXY(MI.Bd, MI.iptoDini, MI.iptoD+1));
						g.draw(pathArrayXY(MI.Bi, MI.iptoIini, MI.iptoI+1));
						if(MI.dist>0) {
							//marcamos el pto mínimo
							g.setStroke(new BasicStroke());
							g.setColor(Color.RED);
							g.draw(new Line2D.Double(pxPtoActual,point2Pixel(ptoRF2Point(MI.indMin))));

							if(MI.iAD<MI.iAI) { //no se han cruzado
								g.setStroke(new BasicStroke(3));
								g.setColor(Color.RED);
								//pintamos rango de puntos en camino
								GeneralPath perimetro = 
									new GeneralPath(GeneralPath.WIND_EVEN_ODD, MI.iAI-MI.iAD+1);

								Point2D.Double px=point2Pixel(ptoRF2Point(MI.iAD));
								perimetro.moveTo((float)px.getX(),(float)px.getY());
								for(int i=MI.iAD+1; i<=MI.iAI; i++ ) {
									px=point2Pixel(ptoRF2Point(i));
									perimetro.lineTo((float)px.getX(),(float)px.getY());
								}
								g.draw(perimetro);
							}
						} else {
							//tenemos libre marcamos punto libre
							g.setStroke(new BasicStroke());
							g.setColor(Color.YELLOW);
							g.draw(new Line2D.Double(pxPtoActual
									,point2Pixel(MI.Tr[MI.iLibre])));

						}
						if(!Double.isInfinite(MI.distCamino) && MI.indSegObs!=Integer.MAX_VALUE) {
							//tenemos los índices
							g.setStroke(new BasicStroke(3));
							g.setColor(Color.GREEN);
							g.draw(pathArrayXY(MI.Tr, MI.indiceCoche
									, MI.indSegObs+1));
							g.draw(new Line2D.Double(point2Pixel(MI.Bi[MI.indSegObs])
									,point2Pixel(MI.Bd[MI.indSegObs])));
							g.draw(new Line2D.Double(point2Pixel(MI.Bi[MI.indiceCoche])
									,point2Pixel(MI.Bd[MI.indiceCoche])));
							if(MI.indBarrSegObs!=Integer.MAX_VALUE) {
								//marcamos pto barrido dió obstáculo camino más cercano
								g.setStroke(new BasicStroke());
								g.draw(new Line2D.Double(pxPtoActual
										,point2Pixel(ptoRF2Point(MI.indBarrSegObs))));
							}
						}
					}
					

				}
			}

		};
		JPanelGrafico.setMinimumSize(new Dimension(400,400));
		JPanelGrafico.setSize(new Dimension(400,400));
		JPanelGrafico.setBackground(Color.BLACK);
		JPanelGrafico.addMouseListener(this);
		add(JPanelGrafico,BorderLayout.CENTER);
		
		JLabel jla=null;
		Border blackline = BorderFactory.createLineBorder(Color.black);
		{ //Parte inferior
			JPanel jpSur=new JPanel();
			
			jla=jlDistLin=new JLabel("   ??.???");
		    Font Grande = jla.getFont().deriveFont(20.0f);
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Dist lineal"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jla.setMinimumSize(new Dimension(300, 20));
			jla.setPreferredSize(new Dimension(130, 45));
			jpSur.add(jla);
			
			jla=jlDistCam=new JLabel("   ??.???");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Dist Camino"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jla.setMinimumSize(new Dimension(300, 20));
			jla.setPreferredSize(new Dimension(130, 45));
			jpSur.add(jla);

			jla=jlFuera=new JLabel("FUERA DEL CAMINO");
		    jla.setFont(Grande);
		    jla.setForeground(Color.RED);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpSur.add(jla);

			add(jpSur,BorderLayout.SOUTH);
		}
	}

	/**
	 * Obtiene posición real de una medida del RF
	 * @param i indice del barrido a considerar
	 * @return posición real obtenidad a partir de posición actual y rumbo en {@link #MI}
	 */
	protected Point2D ptoRF2Point(int i) {
		double ang=MI.Yaw+MI.barr.getAngulo(i)-Math.PI/2;
		double dist=MI.barr.getDistancia(i);
		return new Point2D.Double(MI.posActual[0]+dist*Math.cos(ang),MI.posActual[1]+dist*Math.sin(ang));
	}


	/**
	 * Genera {@link GeneralPath} con puntos en array
	 * @param v array de al menos 2 columnas. La primera se considera coordenada X, la segunda la Y
	 * @param iini indice del primer punto
	 * @param ifin indice siguiente del último punto
	 * @return {@link GeneralPath} con los puntos considerados
	 */
	protected GeneralPath pathArrayXY(double [][] v, int iini, int ifin) {
		if(iini<0 || ifin<=iini || v==null || v.length<ifin || v[0].length<2)
			return null;
		GeneralPath perimetro = 
			new GeneralPath(GeneralPath.WIND_EVEN_ODD, ifin-iini);

		Point2D.Double px=point2Pixel(v[iini][0],v[iini][1]);
		perimetro.moveTo((float)px.getX(),(float)px.getY());
		for(int i=iini+1; i<ifin; i++) {
			px=point2Pixel(v[i][0],v[i][1]);
			//Siguientes puntos son lineas
			perimetro.lineTo((float)px.getX(),(float)px.getY());
		}
		return perimetro;
	}

	/** @return Ídem que {@link #pathArrayXY(double[][], int, int)} usando todo el array.	 */
	protected GeneralPath pathArrayXY(double[][] v) {
		if(v==null)
			return null;
		return pathArrayXY(v, 0, v.length);
		
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
     * Sólo nos interesan pulsaciones del boton 1. 
     * Con SHIFT para determinar posición y orientación. Sin nada para hacer zoom.
     * @see #mouseReleased(MouseEvent)
     */
	public void mousePressed(MouseEvent even) {
		evenPulsa=null;
		evenPos=null;
		if(even.getButton()==MouseEvent.BUTTON1 && (even.getModifiersEx()&MouseEvent.SHIFT_DOWN_MASK)!=0) {
			//Punto del coche
			Point2D.Double nuevaPos=pixel2Point(even.getX(),even.getY());
			System.out.println("Pulsado Boton 1 con shift "+even.getButton()
					+" en posición: ("+even.getX()+","+even.getY()+")"
					+"  ("+nuevaPos.getX()+","+nuevaPos.getY()+")  "
			);
			evenPos=even;
			return;
		}
		if(even.getButton()==MouseEvent.BUTTON1) {
			Point2D.Double ptPulsa=pixel2Point(even.getX(),even.getY());
			System.out.println("Pulsado Boton "+even.getButton()
					+" en posición: ("+even.getX()+","+even.getY()+")"
					+"  ("+ptPulsa.getX()+","+ptPulsa.getY()+")  "
					+" distancia: "+ptPulsa.distance(new Point2D.Double(0,0))
			);
			evenPulsa = even;
			return;
		}
		if(even.getButton()==MouseEvent.BUTTON3) {
			System.out.println("Pulsado Boton "+even.getButton()+" petimos los cálculos");
			MI.masCercano(MI.posActual, MI.Yaw, MI.barr);

			actualiza();
			System.out.println(MI);
			return;
		}
	}

	/**
     * Sólo nos interesan pulsaciones del boton 1. 
     * Con SHIFT para determinar posición y orientación. Sin nada para hacer zoom.
     * Termina el trabajo empezado en {@link #mousePressed(MouseEvent)}
     */
	public void mouseReleased(MouseEvent even) {
		if(even.getButton()==MouseEvent.BUTTON1) {
			System.out.println("Soltado Boton "+even.getButton()
					+" en posición: ("+even.getX()+","+even.getY()+")");
			if ( (even.getModifiersEx()&MouseEvent.SHIFT_DOWN_MASK)!=0
					&& evenPos!=null) {
				System.out.println("Soltado con Shift Boton "+even.getButton()
						+" en posición: ("+even.getX()+","+even.getY()+")");
				//Creamos rectángulo si está suficientemente lejos
				if(Math.abs(even.getX()-evenPos.getX())>50 
						|| Math.abs(even.getY()-evenPos.getY())>50) {
					Point2D.Double nuevaPos=pixel2Point(evenPos.getX(),evenPos.getY());
					Point2D.Double posAngulo=pixel2Point(even.getX(),even.getY());
					double[] npos={nuevaPos.getX(),nuevaPos.getY()};
					{
						BarridoAngular barAct=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
						double frec=(13.6+2*Math.random());
						double Amp=(3.0+15*Math.random());
						double Dpor=(20.0+15*Math.random());
						for(int i=0;i<barAct.numDatos();i++) {
//							barAct.datos[i]=(short)((15.0)*100.0);
							barAct.datos[i]=(short)((Math.sin((double)i/(barAct.numDatos()-1)*Math.PI*frec)
									*Amp
									+Dpor)*100.0);
							//ruido aleatorio
							if(Math.random()<0.05)
								barAct.datos[i]=(short)((Math.random()*60+2)*100);
						}
						MI.masCercano(npos, Math.atan2(nuevaPos.getY()-posAngulo.getY(), nuevaPos.getX()-posAngulo.getX())
								, barAct);
					}
					actualiza();
					System.out.println(MI);

				}
				return;
			}
			if(evenPulsa!=null) {
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


	/**
	 * Devuelve mínimo de entre min y el mínimo de columna ind del vector v.
	 * @param ind columna a comprobar
	 * @param min minimo inicial
	 * @param v vector cuya columna se va a recorrer
	 * @return mínimo de entre min y el mínimo de columna ind del vector v
	 */
	double min(int ind,double min, double[][] v) {
		if(v!=null && v.length>0 && v[0].length>=ind)
			for(int i=0; i<v.length; i++)
				if(v[i][ind]<min)
					min=v[i][ind];
		return min;
	}
	
	/**
	 * @return Mínimo de la columna ind de los 3 vectores pasados
	 */
	double min(int ind, double[][] v1, double[][] v2, double[][] v3) {
		return min(ind,min(ind,min(ind,java.lang.Double.POSITIVE_INFINITY,v1),v2),v3);
	}
	
	/**
	 * Devuelve maximo de entre max y el máximo de columna ind del vector v.
	 * @param ind columna a comprobar
	 * @param max máximo inicial
	 * @param v vector cuya columna se va a recorrer
	 * @return maximo de entre max y el máximo de columna ind del vector v
	 */
	double max(int ind,double max, double[][] v) {
		if(v!=null && v.length>0 && v[0].length>=ind)
			for(int i=0; i<v.length; i++)
				if(v[i][ind]>max)
					max=v[i][ind];
		return max;
	}
	
	/** @return Máximo de la columna ind de los 3 vectores pasados */
	double max(int ind, double[][] v1, double[][] v2, double[][] v3) {
		return max(ind,max(ind,max(ind,java.lang.Double.NEGATIVE_INFINITY,v1),v2),v3);
	}
	

	/**
	 * Acatualiza la presentación con los datos en {@link #MI}.
	 * Se debe invocar cuando {@link #MI} realiza un nuevo cálculo. 
	 */
	public void actualiza() {
		hayDatos=true;
		if(Double.isNaN(MI.dist)) {
			jlDistLin.setEnabled(false);
			jlDistCam.setEnabled(false);
			jlFuera.setEnabled(true);
			
		} else {
			jlFuera.setEnabled(false);
			jlDistLin.setEnabled(true);
			if(MI.dist>0) {
				jlDistLin.setText(String.format("%9.3f m", MI.dist));
				jlDistLin.setForeground(Color.RED);
			} else {
				jlDistLin.setText(String.format("%9.3f m", -MI.dist));
				jlDistLin.setForeground(Color.GREEN);				
			}
			if(MI.indSegObs!=Integer.MAX_VALUE) {
				jlDistCam.setEnabled(true);
				jlDistCam.setText(String.format("%9.3f m", MI.distCamino));				
			} else jlDistCam.setEnabled(false);
		}
		//programamos la actualizacion de la ventana
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				repaint();
			}
		});
	}
	
	
	
	/**
	 * Programa para probar 
	 * @param args
	 */
	public static void main(String[] args) {

		//necestamos leer archivo con la ruta
		Ruta rutaEspacial=null;
		//elegir fichero
		JFileChooser fc=new JFileChooser(new File("./Rutas"));
		do {
			int devuelto=fc.showOpenDialog(null);
			if (devuelto!=JFileChooser.APPROVE_OPTION) 
				JOptionPane.showMessageDialog(null,
						"Necesario cargar fichero de ruta",
						"Error",
						JOptionPane.ERROR_MESSAGE);
			else  {
				String fichRuta=fc.getSelectedFile().getAbsolutePath();
	    		try {
	    			File file = new File(fichRuta);
	    			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
	    			rutaEspacial=(Ruta)ois.readObject();
	    			ois.close();
	    		} catch (IOException ioe) {
	    			System.err.println("Error al abrir el fichero " + fichRuta);
	    			System.err.println(ioe.getMessage());
	    			rutaEspacial=null;
	    		} catch (ClassNotFoundException cnfe) {
	    			System.err.println("Objeto leído inválido: " + cnfe.getMessage());
	    			rutaEspacial=null;
	    		}     
			}
		} while(rutaEspacial==null);

		double [][] Tr=rutaEspacial.toTr();
		System.out.println("Longitud de la trayectoria="+Tr.length);

		JFrame ventana=new JFrame("Panel Mira Obstáculo");
		
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	
		MiraObstaculo mi=new MiraObstaculo(Tr);
		
		PanelMiraObstaculo pmo=new PanelMiraObstaculo(mi);
		
		ventana.add(pmo);
		
		
		//ventana.pack();
		ventana.setSize(new Dimension(800,600));
		ventana.setVisible(true);
		

		//Damos pto, orientación y barrido
		BarridoAngular ba=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
		for(int i=0;i<ba.numDatos();i++) {
//			ba.datos[i]=(short)((15.0)*100.0);
			ba.datos[i]=(short)((Math.sin((double)i/(ba.numDatos()-1)*Math.PI*13.6)*3.0+10.0)*100.0);
		}
		double[] ptoAct={-26, 10};
		double dist=mi.masCercano(ptoAct, Math.toRadians(90), ba);
		pmo.actualiza();
		System.out.println(mi);
		boolean Caminar=false;
		if(Caminar) {
			//vamos recorriendo la trayectoria con barridos aleatorios
			int inTr=10, inTrAnt=8;
			while(true) {
				BarridoAngular barAct=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
				double frec=(13.6+2*Math.random());
				double Amp=(3.0+15*Math.random());
				double Dpor=(20.0+15*Math.random());
				for(int i=0;i<barAct.numDatos();i++) {
//					barAct.datos[i]=(short)((15.0)*100.0);
					barAct.datos[i]=(short)((Math.sin((double)i/(barAct.numDatos()-1)*Math.PI*frec)
							*Amp
							+Dpor)*100.0);
				}

				double diAct=mi.masCercano(Tr[inTr]
				                               , Math.atan2(Tr[inTr][1]-Tr[inTrAnt][1],Tr[inTr][0]-Tr[inTrAnt][0]), barAct);
				System.out.println("Indice "+inTr+" distancia "+diAct);
				pmo.actualiza();
				try {
					Thread.sleep(200);
				} catch (Exception e) { }
				inTrAnt=inTr;
				inTr=(inTr+3)%Tr.length;
			}
		}
	}

}
