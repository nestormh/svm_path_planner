package sibtra.rfyruta;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
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
import javax.swing.border.Border;

import sibtra.gps.Ruta;
import sibtra.gps.Trayectoria;
import sibtra.lms.BarridoAngular;
import sibtra.util.PanelFlow;
import sibtra.util.PanelMuestraTrayectoria;

/**
 * Clase amiga de {@link MiraObstaculo} que muestar graficamente resultado de sus cálculos.
 * Ponemos eje X vertical hacia arriba (Norte) y eje Y horizontal a la izda (Oeste)
 * @author alberto
 */
@SuppressWarnings("serial")
public class PanelMiraObstaculo extends PanelMuestraTrayectoria {
	
	/** Objeto {@link MiraObstaculo} del cual se obtiene toda la información a representar*/
	protected MiraObstaculo MI;
	
	/** Barrido angular de {@link #MI} o fijado expresamente */
	protected BarridoAngular barrAng=null;
	
	/** etiqueta que muestar la distancia al obstaculo */
	protected JLabel jlDistLin;
	/** etiqueta que muestar la distancia al obstaculo */
	protected JLabel jlDistCam;

	/** Etiqueta para indicar cuando estamos fuera del camino */
	protected JLabel jlFuera;

    /**
     * Constructor 
     * @param miObs Obejto {@link MiraObstaculo} del cual se obtendrá toda la información.
     */
	public PanelMiraObstaculo(MiraObstaculo miObs) {
		super();
		
		JLabel jla=null;
		Border blackline = BorderFactory.createLineBorder(Color.black);
		{//nuevo panel para añadir debajo
			JPanel jpPre=new PanelFlow();
			
			//TODO usar labels dato
			jla=jlDistLin=new JLabel("   ??.???");
		    Font Grande = jla.getFont().deriveFont(20.0f);
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Dist lineal"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
//			jla.setMinimumSize(new Dimension(300, 20));
			jla.setPreferredSize(new Dimension(130, 45));
			jpPre.add(jla);
			
			jla=jlDistCam=new JLabel("   ??.???");
			jla.setBorder(BorderFactory.createTitledBorder(
				       blackline, "Dist Camino"));
		    jla.setFont(Grande);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
//			jla.setMinimumSize(new Dimension(300, 20));
			jla.setPreferredSize(new Dimension(130, 45));
			jpPre.add(jla);

			jla=jlFuera=new JLabel("FUERA DEL CAMINO");
		    jla.setFont(Grande);
		    jla.setForeground(Color.RED);
			jla.setHorizontalAlignment(JLabel.CENTER);
			jla.setEnabled(false);
			jpPre.add(jla);

//			jpPre.setMinimumSize(new Dimension(Short.MAX_VALUE,60));
			add(jpPre);

		}
		setMiraObstaculo(miObs);
	}
	
	public void setMiraObstaculo(MiraObstaculo mi) {
		MI=mi;
		if(MI!=null) setTrayectoria(MI.tray);
		actualiza();
	}

	protected void cosasAPintar(Graphics g0) {
		super.cosasAPintar(g0);
		Graphics2D g=(Graphics2D)g0;
		if (posXCoche!=Double.NaN && barrAng!=null) { //pintamos el barrido si existe y estamos situados
			g.setStroke(new BasicStroke());
			g.setColor(Color.WHITE);
			//pintamos rango de puntos en camino
			GeneralPath perimetro = 
				new GeneralPath(GeneralPath.WIND_EVEN_ODD, barrAng.numDatos());

			Point2D.Double px=point2Pixel(ptoRF2Point(0));
			perimetro.moveTo((float)px.getX(),(float)px.getY());
			for(int i=1; i<barrAng.numDatos(); i++ ) {
				px=point2Pixel(ptoRF2Point(i));
				perimetro.lineTo((float)px.getX(),(float)px.getY());
			}
			g.draw(perimetro);
			//Si esta seleccionado puntos, admeás ponemo una cruz en cada punto del barrido.
			if(jcbMostrarPuntos.isSelected())
				// Sacado de puntosArray(g,MI.Bi);
				//pintamos los puntos que están dentro del recuadro
				for(int i=1; i<barrAng.numDatos(); i++ ) {
					Point2D pa=ptoRF2Point(i);
					if(pa.getX()<=esqSI.getX() && pa.getX()>=esqID.getX()
							&& pa.getY()<=esqSI.getY() && pa.getY()>=esqID.getY() ) {
						//esta dentro del recuadro
						Point2D pto=point2Pixel(pa);
						int x=(int)pto.getX(), y=(int)pto.getY();
						g.drawLine(x-tamCruz, y-tamCruz
								, x+tamCruz, y+tamCruz);
						g.drawLine(x-tamCruz, y+tamCruz
								, x+tamCruz, y-tamCruz);
					}
			}
		}
		if(MI==null) return; //si no hay MI, no hacemos nada más 
		g.setStroke(new BasicStroke());
		//pintamos el borde derecho
		g.setColor(Color.BLUE);
		if(!jcbMostrarPuntos.isSelected()) {
			GeneralPath gptr=pathArrayXY(MI.Bd);
			if(gptr!=null) g.draw(gptr);
		} else puntosArray(g,MI.Bd);
		//pintamos el borde izquierdo
		g.setColor(Color.RED);
		if(!jcbMostrarPuntos.isSelected()) {
			GeneralPath gptr=pathArrayXY(MI.Bi);
			if(gptr!=null) g.draw(gptr);
		} else puntosArray(g,MI.Bi);
		if(MI.hayDatos) {
			//pixeles del pto actual se usa para trazar varias líneas
			Point2D pxPtoActual=point2Pixel(MI.posActual);

			//vemos si hay información de colisión
			if(!Double.isNaN(MI.getDistanciaLineal())) {
				//estamos dentro del camino
				//Lineas indicando de donde empezó el barrido
				g.setStroke(new BasicStroke());
				g.setColor(Color.GRAY);
				if(MI.encontradoInicioD) {
				g.draw(new Line2D.Double(pxPtoActual
						,point2Pixel(MI.Bd[MI.iptoDini])));
				g.draw(new Line2D.Double(pxPtoActual
						,point2Pixel(MI.Bd[MI.iptoD])));
				}
				if(MI.encontradoInicioI) {
				g.draw(new Line2D.Double(pxPtoActual
						,point2Pixel(MI.Bi[MI.iptoIini])));
				g.draw(new Line2D.Double(pxPtoActual
						,point2Pixel(MI.Bi[MI.iptoI])));
				}


				g.setStroke(new BasicStroke(2));
				g.setColor(Color.WHITE);
				//los de la derecha e izquierda que están libres
				GeneralPath gp=null;
				if(MI.encontradoInicioD && (gp=pathArrayXY(MI.Bd, MI.iptoDini, MI.iptoD+1,MI.tray.esCerrada()))!=null) g.draw(gp);
				if(MI.encontradoInicioI && (gp=pathArrayXY(MI.Bi, MI.iptoIini, MI.iptoI+1,MI.tray.esCerrada()))!=null) g.draw(gp);
				if(MI.getDistanciaLineal()>0) {
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
				if(!Double.isInfinite(MI.getDistanciaCamino()) && MI.indSegObs!=Integer.MAX_VALUE) {
					//tenemos los índices
					g.setStroke(new BasicStroke(3));
					g.setColor(Color.GREEN);
					if((gp=pathArrayXY(MI.Tr, MI.indiceCoche
							, MI.indSegObs+1, MI.tray.esCerrada()))!=null) g.draw(gp);
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

	
	/**
	 * Obtiene posición real de una medida del RF
	 * @param i indice del barrido a considerar
	 * @return posición real obtenidad a partir de posición actual y rumbo
	 */
	protected Point2D ptoRF2Point(int i) {
		double ang=orientacionCoche+barrAng.getAngulo(i)-Math.PI/2;
		double dist=barrAng.getDistancia(i);
		return new Point2D.Double(posXCoche+dist*Math.cos(ang),posYCoche+dist*Math.sin(ang));
	}

	/** Fija posición, orientación y barrido y actualiza. Cualquiera de ellos puede ser null */
	public void setPosicionYawBarrido(double[] pos, double yaw, BarridoAngular barr) {
		if(pos!=null && pos.length<2 )
			throw new IllegalArgumentException("La posición del coche debe ser array de, al menos, 2 valores");
		situaCoche(pos[0], pos[1], yaw);
		barrAng=barr;
		actualiza();
	}
	

	/**
	 * Acatualiza la presentación con los datos en {@link #MI}.
	 * Se debe invocar cuando {@link #MI} realiza un nuevo cálculo. 
	 */
	public void actualiza() {
		if(MI!=null && MI.hayDatos) {
			situaCoche(MI.posActual[0], MI.posActual[1], MI.Yaw);
			barrAng=MI.barr;
		}
		if(MI==null || Double.isNaN(MI.getDistanciaLineal())) {
			jlDistLin.setEnabled(false);
			jlDistCam.setEnabled(false);
			jlFuera.setEnabled(true);
			
		} else {
			jlFuera.setEnabled(false);
			jlDistLin.setEnabled(true);
			if(MI.getDistanciaLineal()>0) {
				jlDistLin.setText(String.format("%9.3f m", MI.getDistanciaLineal()));
				jlDistLin.setForeground(Color.RED);
			} else {
				jlDistLin.setText(String.format("%9.3f m", -MI.getDistanciaLineal()));
				jlDistLin.setForeground(Color.GREEN);				
			}
			if(MI.indSegObs!=Integer.MAX_VALUE) {
				jlDistCam.setEnabled(true);
				jlDistCam.setText(String.format("%9.3f m", MI.getDistanciaCamino()));				
			} else jlDistCam.setEnabled(false);
		}
		super.actualiza();
	}
	

public static Ruta leeRutaEspacialDeFichero(String fichRuta) {
	Ruta rutaEspacial;
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
	return rutaEspacial;
}
	
	
	/**
	 * Programa para probar.
	 * Sin argumentos es interactivo y no comanina. El vehiculo se posiciona con el ratón.
	 * Primer argumento número de iteraciones, el vehiculo camina el número de iteraciones indicadas. Si es <=0 no camina.
	 * Segundo argumento fichero de ruta, ya no es interactivo.
	 * @param args
	 */
	public static void main(String[] args) {

		//necestamos leer archivo con la ruta
		boolean esInteractivo=true;
		boolean Caminar=false;

		int numIteras=1000; //para el caso de que no sea interactivo
		//primer argumento, número de iteracines
		if(args.length>=1) {
			try {
				numIteras=Integer.parseInt(args[0]);
			} catch (NumberFormatException e) {
				System.err.println("Primer argumneto no se entiende como entero (número de iteraciones)");
				System.exit(1);
			}
			Caminar=numIteras>0;
		}
		Ruta rutaEspacial=null;
		if(args.length<2) { //no se ha pasado 2º argumento
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
					rutaEspacial=leeRutaEspacialDeFichero(fc.getSelectedFile().getAbsolutePath());
				}
			} while(rutaEspacial==null);
		} else {
			//nombre de fichero se pasa en linea de comandos
			if((rutaEspacial=leeRutaEspacialDeFichero(args[1]))==null) {
				System.err.println("No se ha podido leer fichero de ruta :"+args[1]);
				System.exit(1);
			}
			esInteractivo=!Caminar; //Si no camina, será interactivo
		}
		
		Trayectoria Tr=new Trayectoria(rutaEspacial);
		System.out.println("Longitud de la trayectoria="+Tr.length()
				+" es cerrad="+Tr.esCerrada());

		MiraObstaculo mi=new MiraObstaculo(Tr);
		PanelMiraObstaculo pmo=null;
		if(esInteractivo) {
			JFrame ventana=new JFrame("Panel Mira Obstáculo");
			ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			pmo=new PanelMiraObstaculo(mi) {
				/** Evento cuando se pulsó el ratón con el SHIFT, establece la posición deseada */
				MouseEvent evenPos;
				Point2D.Double nuevaPos;
				Point2D.Double posAngulo;

				/**
				 * Sólo nos interesan pulsaciones del boton 1. 
				 * Con CONTROL para determinar posición y orientación. Sin nada para hacer zoom.
				 * @see #mouseReleased(MouseEvent)
				 */
				public void mousePressed(MouseEvent even) {
					evenPos=null;
					if(even.getButton()==MouseEvent.BUTTON1 && (even.getModifiersEx()&MouseEvent.CTRL_DOWN_MASK)!=0) {
						//Punto del coche
						Point2D.Double nuevaPos=pixel2Point(even.getX(),even.getY());
						System.out.println("Pulsado Boton 1 con CONTROL "+even.getButton()
								+" en posición: ("+even.getX()+","+even.getY()+")"
								+"  ("+nuevaPos.getX()+","+nuevaPos.getY()+")  "
						);
						evenPos=even;
						return;
					}
					if(even.getButton()==MouseEvent.BUTTON3) {
						System.out.println("Pulsado Boton "+even.getButton()+" pedimos los cálculos");
						MI.masCercano(MI.Yaw, MI.barr);

						actualiza();
						System.out.println(MI);
						return;
					}
					//al del padre lo llamamos al final
					super.mousePressed(even);
				}

				/**
				 * Las pulsaciones del boton 1 con CONTROL para determinar posición y orientación.
				 * Termina el trabajo empezado en {@link #mousePressed(MouseEvent)}
				 */
				public void mouseReleased(MouseEvent even) {
					if(even.getButton()==MouseEvent.BUTTON1 
							&& (even.getModifiersEx()&MouseEvent.CTRL_DOWN_MASK)!=0
							&& evenPos!=null) {
						System.out.println("Soltado con Control Boton "+even.getButton()
								+" en posición: ("+even.getX()+","+even.getY()+")");
						//Cambiamos la posición si el movimiento es suficientemente grande
						if(Math.abs(even.getX()-evenPos.getX())>50 
								|| Math.abs(even.getY()-evenPos.getY())>50) {
							nuevaPos=pixel2Point(evenPos.getX(),evenPos.getY());
							posAngulo=pixel2Point(even.getX(),even.getY());
						}
						//Aunque no haya nueva posición, hacemos nuevo barrido
						double[] npos={nuevaPos.getX(),nuevaPos.getY()};
						BarridoAngular barAct=new BarridoAngular(181,0,4,(byte)2,false,(short)2); //con true mm, false cm
						double frec=(13.6+2*Math.random());
						double Amp=(3.0+15*Math.random());
						double Dpor=(15.0+15*Math.random());
						for(int i=0;i<barAct.numDatos();i++) {
//							barAct.datos[i]=(short)((15.0)*1000.0);
							barAct.datos[i]=(short)((Math.sin((double)i/(barAct.numDatos()-1)*Math.PI*frec)
									*Amp
									+Dpor)*100.0);
							//ruido aleatorio
							if(Math.random()<0.05)
								barAct.datos[i]=(short)((Math.random()*60+2)*100.0);
						}
						long tini=System.currentTimeMillis();
						MI.tray.situaCoche(npos);
						MI.masCercano(Math.atan2(nuevaPos.getY()-posAngulo.getY(), nuevaPos.getX()-posAngulo.getX())
								, barAct);
						actualiza();
						System.out.println(MI);
						System.out.println("Tarda:"+(System.currentTimeMillis()-tini));
						return;
					} else {
						//Sacamos indice de pto más cercano
						Point2D.Double pos = pixel2Point(even.getX(),even.getY());
						System.out.println("Indice de Tr más cercano:"+MI.tray.indiceMasCercano(pos.getX(), pos.getY()));
					}
					//al final llamamos al del padre
					super.mouseReleased(even);
				}
			};

			ventana.add(pmo);


			//ventana.pack();
			ventana.setSize(new Dimension(800,600));
			ventana.setVisible(true);
		}

		//Damos pto, orientación y barrido
		BarridoAngular ba=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
		for(int i=0;i<ba.numDatos();i++) {
//			ba.datos[i]=(short)((15.0)*100.0);
			ba.datos[i]=(short)((Math.sin((double)i/(ba.numDatos()-1)*Math.PI*13.6)*3.0+10.0)*100.0);
		}
		Tr.situaCoche(-26, 10);
		double dist=mi.masCercano( Math.toRadians(90), ba);
		if(esInteractivo) pmo.actualiza();
		System.out.println(mi);
		if(Caminar) {
			//vamos recorriendo la trayectoria con barridos aleatorios
			int inTr=10, inTrAnt=8;
			int iteracion=0;
			while(iteracion<numIteras) { //si no es interactivo repetimos sólo 1000 veces
				iteracion++;
				BarridoAngular barAct=new BarridoAngular(181,0,4,(byte)2,false,(short)2);
				double frec=(13.6+2*Math.random());
				double Amp=(3.0+15*Math.random());
				double Dpor=(20.0+15*Math.random());
				for(int i=0;i<barAct.numDatos();i++) {
//					barAct.datos[i]=(short)((15.0)*100.0);
					barAct.datos[i]=(short)((Math.sin((double)i/(barAct.numDatos()-1)*Math.PI*frec)
							*Amp
							+Dpor)*100.0);
					//ruido aleatorio
					if(Math.random()<0.05)
						barAct.datos[i]=(short)((Math.random()*60+2)*100);
				}
				long tini=System.currentTimeMillis();
				Tr.situaCoche(Tr.x[inTr],Tr.y[inTr]);
				double diAct=mi.masCercano(Math.atan2(Tr.y[inTr]-Tr.y[inTrAnt],Tr.x[0]-Tr.x[0]), barAct);
				long tfin=System.currentTimeMillis();
				System.out.println(iteracion+"- Indice "+inTr+" distancia "+diAct
						+"  Tarda:"+(tfin-tini));
				if(esInteractivo) {
					pmo.actualiza();
					try {
						Thread.sleep(2000);
					} catch (Exception e) { }
				}
				inTrAnt=inTr;
				inTr=(inTr+3)%Tr.length();
			}
		}
	}

}
