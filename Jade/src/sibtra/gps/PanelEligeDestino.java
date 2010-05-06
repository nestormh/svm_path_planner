/**
 * 
 */
package sibtra.gps;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.io.File;
import java.util.Vector;

import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.Timer;

import sibtra.util.PanelMuestraVariasTrayectorias;

/**
 * Panel que mostrará los distintos tramos.
 * Con el main se permitirá depurar {@link GestionFlota}
 * 
 * @author alberto
 *
 */
public class PanelEligeDestino extends PanelMuestraVariasTrayectorias {

	GestionFlota gesFlota;
	protected boolean habilitadaEleccionDestino;
	protected double[] destino=new double[2];
	protected boolean hayDestino=false;
	
	/** array con los indices de los tramos que constituyen la ruta actual. Null si no hay */
	int[] arrIndRutaAct=null;
	int[] arrIndRutaAnt=null;
	int indTramoDest=0;
	protected Timer timerActulizacion;
	protected JComboBox jcbDestinos=null; 
	
	public PanelEligeDestino(GestionFlota gFlota) {
		super();
		if(gFlota==null)
			throw new IllegalArgumentException("Gestion de flota no puede ser null");
		gesFlota=gFlota;
		
		//añadimos los tramos al panel
		Trayectoria[] arrTr=gesFlota.getTrayectorias();
		
		for(int i=0; i<arrTr.length; i++) {
			if(añadeTrayectoria(arrTr[i])!=i)
				System.err.println("Inconguencia al añadir trayectoria");
		}
		actulizacionPeridodica(500); //cada 500 ms
		
		//Si hay destinos Combo para seleccionar
		Vector<GPSData> vdes=gesFlota.getDestinos();
		if(vdes!=null && vdes.size()>0) {
			jpSur.add(new JLabel("Destino"));
			jcbDestinos=new JComboBox();
			jcbDestinos.addItem("-- Otro --");
			for(GPSData pa: vdes)
				jcbDestinos.addItem(pa.getNombre());
			jpSur.add(jcbDestinos);
			jcbDestinos.setSelectedIndex(0);
			jcbDestinos.addActionListener(this);
		}
	}
	
	/** Marcamos los destinos definidos y el destino seleccionado si hay ruta */
	@Override
	protected void cosasAPintar(Graphics g0) {
		super.cosasAPintar(g0);
		Graphics2D g=(Graphics2D)g0;
		
		Vector<GPSData> vdes=gesFlota.getDestinos();
		if(vdes!=null && vdes.size()>0) {
			//marcamos los posibles destinos
			g.setStroke(strokeGruesa);
			g.setColor(Color.WHITE);
			for(GPSData da: vdes) {
				da.calculaLocales(gesFlota.getCentro());
				double pa[]={da.getXLocal(),da.getYLocal()};
				if(pa[0]<=esqSI.getX() && pa[0]>=esqID.getX()
						&& pa[1]<=esqSI.getY() && pa[1]>=esqID.getY() ) {
					//esta dentro del recuadro
					Point2D px=point2Pixel(pa);
					int x=(int)px.getX(), y=(int)px.getY();
					g.fillOval(x-tamCruz*2, y-tamCruz*2, tamCruz*4, tamCruz*4);
				}
			}
		}

		if(habilitadaEleccionDestino && jcbDestinos!=null)
			jcbDestinos.setEnabled(true);
		
		if(hayDestino) {
			//Marcamos el destino
			g.setStroke(strokeGruesa);
			g.setColor(Color.WHITE);
			if(destino[0]<=esqSI.getX() && destino[0]>=esqID.getX()
					&& destino[1]<=esqSI.getY() && destino[1]>=esqID.getY() ) {
				//esta dentro del recuadro
				Point2D px=point2Pixel(destino);
				int x=(int)px.getX(), y=(int)px.getY();
				g.fillRect(x-tamCruz*2, y-tamCruz*2, tamCruz*4, tamCruz*4);
			}
		}
		
	}



	/** Establece los tramos que forman la ruta */
	public void estableceRuta(int[] arrInd) {
		arrIndRutaAct=arrInd;
	}
	
	
	private void desDestacaTodos() {
		for(int i=0; i<trays.size(); i++)
			setDestacado(i, false); //desmarcamos todos		
	}
	
	/** Establece timer para la actulización periódica del panel.
	 * Si ya existe uno simplemente cambia el periodo.
	 * En la actulización se va rotando por los tramos que forman la ruta actual
	 * @param periodoMili periodo de actulización en milisegundos
	 */
	public void actulizacionPeridodica(int periodoMili) {
		if(timerActulizacion==null) {
			//creamos el action listener y el timer
			ActionListener taskPerformer = new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					if(arrIndRutaAct!=arrIndRutaAnt) {
						//se cambio de ruta
						desDestacaTodos();
						indTramoDest=0; //empezamos por el primero
						arrIndRutaAnt=arrIndRutaAct;
						actualiza();
					}
					if(arrIndRutaAct!=null && arrIndRutaAct.length>0) {
						int l=arrIndRutaAct.length;
						setDestacado(arrIndRutaAct[(indTramoDest+l-1)%l], false); //apagamos el que estaba antes
						setDestacado(arrIndRutaAct[indTramoDest], true);
						actualiza();
						indTramoDest=(indTramoDest+1)%l;
					}
				}
			};
			timerActulizacion=new Timer(periodoMili, taskPerformer);
		} else 
			//basta con modificar el delay
			timerActulizacion.setDelay(periodoMili);
		timerActulizacion.start();
	}

	/** Detiene la acutualización periodica si existe alguna */
	public void actualizacionPeriodicaParar() {
		if(timerActulizacion==null) return;
		timerActulizacion.stop();
	}
	

	/**
     * Click del boton 1 Elige destino se está habilitado
     */
	public void mouseClicked(MouseEvent even) {
		if( (even.getButton()==MouseEvent.BUTTON1) 
				&& ((even.getModifiersEx()&MouseEvent.SHIFT_DOWN_MASK)==0)
				&& ((even.getModifiersEx()&MouseEvent.CTRL_DOWN_MASK)==0) 
			) {
			System.out.println(getClass().getName()+": Clickeado Boton "+even.getButton()
					+" en posición: ("+even.getX()+","+even.getY()+") "
					+even.getClickCount()+" veces");
			if(habilitadaEleccionDestino) {
				Point2D.Double ptPulsa=pixel2Point(even.getX(),even.getY());
				hayDestino=true;
				destino[0]=ptPulsa.getX(); destino[1]=ptPulsa.getY();
				estableceRuta(gesFlota.indicesTramosADestino(posXCoche, posYCoche
						, orientacionCoche, ptPulsa.getX(), ptPulsa.getY()));
				if(jcbDestinos!=null) {
					//Ponemos el destino como OTRO (que es el último)
					jcbDestinos.setSelectedIndex(jcbDestinos.getItemCount()-1);
				}
				actualiza();
			}
		}
	}

	/** Pone como destino lo elegido */
	public void actionPerformed(ActionEvent ae) {
		super.actionPerformed(ae);
		if(ae.getSource()==jcbDestinos && habilitadaEleccionDestino) {
			System.out.println("\n\nTratamos de ir a "+jcbDestinos.getSelectedItem());
			if(jcbDestinos.getSelectedIndex()>0) {
				int indiceDestino=jcbDestinos.getSelectedIndex()-1;
				GPSData pd=gesFlota.getDestinos().get(indiceDestino);
				destino[0]=pd.getXLocal();  destino[1]=pd.getYLocal();
				hayDestino=true;
				estableceRuta(gesFlota.indicesTramosADestino(posXCoche, posYCoche
						, orientacionCoche, indiceDestino));
				actualiza();
			}
		}

	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		GestionFlota gf=new GestionFlota();
		String nomFich="Rutas/Tramos/TodosLosTramos_PrioridadYOposicion.tra";
		if(args.length>0)
			nomFich=args[0];
		gf.cargaTramos(new File(nomFich));
		if(args.length>1) {
			gf.addDestino(ManejaGPX.cargaPuntos(new File(args[1])));
		}
		
		System.out.println(gf.getTramos().toStringDetallado());
		
		JFrame vent=new JFrame("Panel Elige Destino");
		PanelEligeDestino ped=new PanelEligeDestino(gf) {
            /** Evento cuando se pulsó el ratón con el SHIFT, establece la posición deseada */
            MouseEvent evenPos;

            /**
             * Sólo nos interesan pulsaciones del boton 1. 
             * Con CONTROL para determinar posición y orientación. Sin nada para hacer zoom.
             * @see #mouseReleased(MouseEvent)
             */
            public void mousePressed(MouseEvent even) {
                evenPos = null;
                if (even.getButton() == MouseEvent.BUTTON3) {
                    //Punto del coche
                    Point2D.Double nuevaPos = pixel2Point(even.getX(), even.getY());
                    System.out.println("Pulsado Boton " + even.getButton() + " en posición: (" + even.getX() + "," + even.getY() + ")" 
                    		+ "  (" + nuevaPos.getX() + "," + nuevaPos.getY() + ")  ");
                    evenPos = even;
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
                if (even.getButton() == MouseEvent.BUTTON3 && evenPos != null) {
                    System.out.println("Soltado Boton " + even.getButton() + " en posición: (" + even.getX() + "," + even.getY() + ")");
                    //Creamos rectángulo si está suficientemente lejos
                    if (Math.abs(even.getX() - evenPos.getX()) > 50 || Math.abs(even.getY() - evenPos.getY()) > 50) {
                        Point2D.Double nuevaPos = pixel2Point(evenPos.getX(), evenPos.getY());
                        Point2D.Double posAngulo = pixel2Point(even.getX(), even.getY());
                        double yaw = Math.atan2(nuevaPos.getY() - posAngulo.getY(), nuevaPos.getX() - posAngulo.getX());
                        situaCoche(nuevaPos.getX(), nuevaPos.getY(), yaw);
                        estableceRuta(null);
                        System.out.println("Situado coche en  (" + nuevaPos.getX() +","+ nuevaPos.getY()+") con angulo "+ yaw );
                        habilitadaEleccionDestino=true;
                        hayDestino=false;
                        jcbDestinos.setSelectedIndex(0); //ponemos --otro--
                        actualiza();

                    }
                    return;
                }
                //al final llamamos al del padre
                super.mouseReleased(even);
            }
		};
		
		vent.add(ped);
		vent.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		//vent.setBounds(0, 384, 1024, 742);
		vent.pack();
		vent.setVisible(true);

		
	}

}
