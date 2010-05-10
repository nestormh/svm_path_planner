/**
 * 
 */
package sibtra.gps;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.awt.geom.QuadCurve2D;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Vector;

import javax.swing.AbstractAction;
import javax.swing.AbstractCellEditor;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.KeyStroke;
import javax.swing.ListSelectionModel;
import javax.swing.border.Border;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;

import sibtra.util.PanelMuestraVariasTrayectorias;

/**
 * Permite la edición de ficheros de ruta
 * 
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class EditaFicherosRuta extends JFrame implements  ItemListener, ActionListener, ChangeListener, MouseListener {

	protected Tramos tramos=new Tramos();
	
	protected int indRutaActual=-1; //ninguna seleccionada
	
	private PanelExaminaRuta per;
	
	/** Barra de menu ventana principal */
	JMenuBar barraMenu;
	/** Menu de archivo dentro de la ventana principal */
	JMenu menuArchivo;
	/** Boton de salir del menu de archivo */
	private JMenuItem miSalir;

	private JFileChooser fc;
	private PanelMuestraVariasTrayectorias pmvt;

	private ModeloTablaRutas modeloTR;
	private JTable tablaRutas;

	private JLabel jlNomFicheroDestino;
	
	public EditaFicherosRuta(String titulo) {
		
		super(titulo);
		fc=new JFileChooser(new File("./Rutas"));
//		JPanel cp=(JPanel)getContentPane();
		JPanel cp=new JPanel();
		cp.setLayout(new BoxLayout(cp,BoxLayout.PAGE_AXIS));
//		cp.setBorder(BorderFactory.createMatteBorder(3, 3, 3, 3, Color.YELLOW));
		
		
		pmvt=new PanelMuestraVariasTrayectorias() {
			
			BasicStroke strokeGruesaDiscontinua=new BasicStroke(2.0f
					,BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND, 1.0f
					,new float[] {20f,5f,20f,5f  ,5f,5f ,5f,5f ,5f,5f ,5f,5f ,5f,5f ,5f,5f}
			,0f);

			protected void cosasAPintar(Graphics g0) {
				super.cosasAPintar(g0);
				Graphics2D g=(Graphics2D)g0;
//				if(indRutaActual<0) return; //no añadimos nada
				//marcamos siguientes de la ruta actual
				g.setStroke(strokeGruesaDiscontinua);
				for(int indIni=0; indIni<tramos.size();indIni++)
					for(int isig=0; isig<tramos.size(); isig++) {
						//Indicamos tramo Siguiente
						if(tramos.isSiguiente(indIni, isig)) {
							g.setColor(Color.GREEN);
							//punto inicial, ultimo del actual
							GPSData ptoIni=tramos.getRuta(indIni).getUltimoPto();
							Point2D.Double pxIni=point2Pixel(ptoIni.getXLocal(), ptoIni.getYLocal());
							//punto final, primero del siguiente
							GPSData ptoFin=tramos.getRuta(isig).getPunto(0);
							Point2D.Double pxFin=point2Pixel(ptoFin.getXLocal(), ptoFin.getYLocal());
							double vx=pxFin.x-pxIni.x;
							double vy=pxFin.y-pxIni.y;
							//es siguiente, lo marcamos
							g.draw(new QuadCurve2D.Double(
									pxIni.x,pxIni.y
									,pxFin.x+4*vx+4*vy,pxFin.y+4*vy-4*vx
									,pxFin.x, pxFin.y
							));
						}
						//Indicamos tramos prioritario
						if(tramos.isPrioritatio(indIni, isig)) {
							//tiene prioridad
							g.setColor(Color.RED);
							//punto inicial, ultimo del actual
							Ruta ra=tramos.getRuta(indIni);
							GPSData ptoIni=ra.getPunto(ra.getNumPuntos()-3>0?ra.getNumPuntos()-3:0);
							Point2D.Double pxIni=point2Pixel(ptoIni.getXLocal(), ptoIni.getYLocal());
							//punto final, primero del siguiente
							ra=tramos.getRuta(isig);
							GPSData ptoFin=ra.getPunto(ra.getNumPuntos()-3>0?ra.getNumPuntos()-3:0);
							Point2D.Double pxFin=point2Pixel(ptoFin.getXLocal(), ptoFin.getYLocal());
							double vx=pxFin.x-pxIni.x;
							double vy=pxFin.y-pxIni.y;
							//es siguiente, lo marcamos
							g.draw(new QuadCurve2D.Double(
									pxIni.x,pxIni.y
									,pxFin.x+4*vy,pxFin.y-4*vx
									,pxFin.x, pxFin.y
							));
//							g.draw(new Line2D.Double(pxIni.x,pxIni.y,pxFin.x, pxFin.y));
						}
						//Indicamos tramos prioritario en oposicion
						if(tramos.isPrioritarioOposicion(indIni, isig)) {
							//tiene prioridad en oposicion
							//marcamos entre puntos centrales de ambos tramos
							g.setColor(Color.RED);
							//punto inicial, el punto medio de la ruta actual
							Ruta ra=tramos.getRuta(indIni);
							GPSData ptoIni=ra.getPunto(ra.getNumPuntos()/2);
							Point2D.Double pxIni=point2Pixel(ptoIni.getXLocal(), ptoIni.getYLocal());
							//punto final, el medio de la otra ruta
							ra=tramos.getRuta(isig);
							GPSData ptoFin=ra.getPunto(ra.getNumPuntos()/2);
							Point2D.Double pxFin=point2Pixel(ptoFin.getXLocal(), ptoFin.getYLocal());
							GeneralPath gp=new GeneralPath();
							gp.moveTo(pxIni.x, pxIni.y);
							double tamVec=200.0/(Double.isNaN(getEscala())?1:getEscala());
							gp.lineTo(pxIni.x+tamVec*Math.sin(ptoIni.getAngulo()), pxIni.y-tamVec*Math.cos(ptoIni.getAngulo()));
							gp.lineTo(pxFin.x+tamVec*Math.sin(ptoFin.getAngulo()), pxFin.y-tamVec*Math.cos(ptoFin.getAngulo()));							
							gp.lineTo(pxFin.x, pxFin.y);
							g.draw(gp);
						}
					}
			}
		};
		pmvt.getJPanelGrafico().addMouseListener(this); //para recibir el ratón
		pmvt.setSeguirCoche(false);
		pmvt.setMostrarCoche(false);
		pmvt.setEscala(50);
//		pmr.setPreferredSize(new Dimension(Integer.MAX_VALUE,Integer.MAX_VALUE));
//		pmr.setMinimumSize(new Dimension(Integer.MAX_VALUE,Integer.MAX_VALUE));
		
		//Bajo el mapa solapas con rutas mostradas y punto
		JTabbedPane panSolapas=new JTabbedPane(JTabbedPane.TOP,JTabbedPane.WRAP_TAB_LAYOUT);
		//Solapa con tabla con datos de las rutas
		modeloTR=new ModeloTablaRutas();
		tablaRutas=new JTable(modeloTR);
		tablaRutas.setPreferredScrollableViewportSize(new Dimension(500, 70));
		tablaRutas.setColumnSelectionAllowed(false);
		tablaRutas.setRowSelectionAllowed(true);
		tablaRutas.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		//para representación y edición del color
		tablaRutas.setDefaultRenderer(Color.class, new ColorRenderer(true));
		tablaRutas.setDefaultEditor(Color.class, new ColorEditor());

        ajustaAnchos();

		panSolapas.addTab("Rutas", new JScrollPane(tablaRutas));

		//Panel con datos de los puntos
		per=new PanelExaminaRuta();
		panSolapas.addTab("Punto",new JScrollPane(per));		

		//Panel para el fichero de destinos
		JPanel panelFichDestino=new JPanel();
		jlNomFicheroDestino=new JLabel("Fichero Destinos:");
		panelFichDestino.add(jlNomFicheroDestino);
		AccionEligeFicheroDestinos accionFicheroDestino = new AccionEligeFicheroDestinos();
		panelFichDestino.add(new JButton(accionFicheroDestino));
		panSolapas.addTab("Destinos", panelFichDestino);
		
		//split panel en el centro de la ventana principal
        JSplitPane splitPanel=new JSplitPane(JSplitPane.VERTICAL_SPLIT
//        		,false  //si al mover la barra componentes se refrescan continuamente
        		,true  //si al mover la barra componentes se refrescan continuamente
        		,pmvt
        		,panSolapas
        );
        splitPanel.setOneTouchExpandable(true);
//        splitPanel.setBorder(BorderFactory.createMatteBorder(3, 3, 3, 3, Color.GREEN));
//		splitPanel.setPreferredSize(new Dimension(Integer.MAX_VALUE,Integer.MAX_VALUE));
//		splitPanel.setMinimumSize(new Dimension(Integer.MAX_VALUE,Integer.MAX_VALUE));
        splitPanel.setAlignmentX(Component.CENTER_ALIGNMENT);
        cp.add(splitPanel);

        //barra de menu
        barraMenu=new JMenuBar();
        setJMenuBar(barraMenu); //ponemos barra en la ventana
        //menu de archivo
        menuArchivo=new JMenu("Fichero");
        barraMenu.add(menuArchivo);

        menuArchivo.add(new AccionCargarRuta())
          .setAccelerator(KeyStroke.getKeyStroke( KeyEvent.VK_O,KeyEvent.CTRL_MASK));

        menuArchivo.add(new AccionCargarTramos())
        .setAccelerator(KeyStroke.getKeyStroke( KeyEvent.VK_T,KeyEvent.CTRL_MASK));

        menuArchivo.addSeparator();
        
        menuArchivo.add(new AccionSalvarTramos())
        .setAccelerator(KeyStroke.getKeyStroke( KeyEvent.VK_S,KeyEvent.CTRL_MASK));

        menuArchivo.addSeparator();
        
        menuArchivo.add(accionFicheroDestino);
        
        miSalir=new JMenuItem("Salir");
        miSalir.setAccelerator(KeyStroke.getKeyStroke( KeyEvent.VK_Q,KeyEvent.CTRL_MASK));
        miSalir.addActionListener(this);
        menuArchivo.add(miSalir);
        //lo añadimos al menu al final

		
		per.jsDato.addChangeListener(this);
		
		setContentPane(cp);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(0, 384, 1024, 742);
		pack();
		setVisible(true);
	}


	/**
	 * @param tablaRutas
	 */
	private void ajustaAnchos() {
		//fijamos tamaños preferidos
        for(int i=0; i<modeloTR.getColumnCount();i++)
        	tablaRutas.getColumnModel().getColumn(i).setPreferredWidth(
        			modeloTR.getLargoColumna(i)	        			
        	);
        tablaRutas.repaint();
	}


	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==miSalir) {
			Terminar();
			return;
		}
//		if(e.getSource()==jmiAcercaDe) {
//		JOptionPane.showMessageDialog(ventanaPrincipal, "VERDINO: ISAATC ULL");
//		}
	}

	private void añadeRuta(Ruta ruta, String nombre ){
		tramos.añadeTramo(ruta, nombre);
		if(pmvt.añadeTrayectoria(new Trayectoria(ruta,Double.MAX_VALUE,-1))!=(tramos.size()-1) )
			throw new IllegalStateException("Trayectoria no tendrá el mismo índice en el panel");
		ajustaAnchos();
	}
	
	/** Accinones (no gráficas) a tomar al borrar una trayectoria
	 * @param indice a borrar, normalmente será {@link #indRutaActual}
	 */
	private void borraTrayectoria(int ind) {
		tramos.borraTramo(ind);
		pmvt.borraTrayectoria(ind);
	}
	
	/**
	 * @param file Fichero de Ruta a cargar
	 */
	protected void cargaFicheroRuta(File file) {
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			Ruta ruta=(Ruta)ois.readObject();
			añadeRuta(ruta, file.getName()+"_Espacial");
			ruta=(Ruta)ois.readObject();
			añadeRuta(ruta,file.getName()+"_Temporal");
			ois.close();
			
		} catch (IOException ioe) {
			System.err.println("Error al abrir el fichero " + file.getName());
			System.err.println(ioe.getMessage());
		} catch (ClassNotFoundException cnfe) {
			System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
		}
		modeloTR.actualiza();
		pmvt.actualiza();
	}
	
    /** Metodo para terminar la ejecución */
    protected void Terminar() {
    	System.out.println("Terminamos ...");
		System.exit(0);
	}

    /** Se ha elegido fichero de destinos */
	protected void estableceFicheroDestino(File file) {
		//Tratamos de cargar el fichero
		Vector<GPSData> vdes=ManejaGPX.cargaPuntos(file);
		if(vdes==null) {
//			custom title, error icon
			JOptionPane.showMessageDialog(this,
			    "No se pudo cargar puntos del fichero"+file.getName(),
			    "Error de lectura",
			    JOptionPane.ERROR_MESSAGE);
			return;
		}
		//Ponemos los destinos en el panel.
		double[][] arrDest=null;
		if(vdes!=null && vdes.size()>0) {
			arrDest=new double[vdes.size()][2];
			for(int i=0;i<vdes.size();i++) {
				GPSData pa=vdes.get(i);
				pa.calculaLocales(tramos.getCentro());
				arrDest[i][0]=pa.getXLocal();
				arrDest[i][1]=pa.getYLocal();
			}
		}
		//añadimos los destinos al panel
		pmvt.setDestinos(arrDest);
		pmvt.actualiza();
		//Buscamos el nombre relativo del fichero
		String nomFich=null;
		System.out.println("Path a fichero destino "+ file.getAbsolutePath()
				+" user.dir="+System.getProperty("user.dir"));
		if(!file.getAbsolutePath().startsWith(System.getProperty("user.dir"))) {
			System.err.println("Path a fichero destino "+ file.getAbsolutePath()
					+" no comienza por user.dir="+System.getProperty("user.dir"));
			nomFich=file.getName();
		} else { 
			//quitamos directorio actual (y la barra) del path absoluto
			nomFich=file.getAbsolutePath().substring(System.getProperty("user.dir").length()+1); 
			if(tramos!=null)
				tramos.setNombFichDestinos(nomFich);
			System.out.println("Establecemos fichero de destinos '"+nomFich+"' en tramos");
		}
		jlNomFicheroDestino.setText("Nombre fichero destinos: "+nomFich);
	}

    
	public void itemStateChanged(ItemEvent e) {
	}
	
	/** Para gestionar los cambios en el punto seleccionado en el {@link #per} */
	public void stateChanged(ChangeEvent arg0) {
		GPSData ultPto=per.ruta.getPunto((Integer)per.jsDato.getValue());
		if(ultPto==null)
			return;
		double x=ultPto.getXLocal();
		double y=ultPto.getYLocal();
		double yaw=ultPto.getAngulo();
		if(ultPto.getAngulosIMU()!=null)
			yaw=Math.toRadians(ultPto.getAngulosIMU().getYaw());
		pmvt.situaCoche(x, y, yaw);
		pmvt.actualiza();
	}
	
	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}
	
	/** 
	 * Si es pulsación 1 seleccionamos punto más cercano para representar
	 * Si es 3 mostramos menu del punto.
	 * Sólo si no hay modificadores.
	 * @param even evento
	 */
	public void mousePressed(MouseEvent even) {
		if( ((even.getModifiersEx()&MouseEvent.SHIFT_DOWN_MASK)!=0)
			|| ((even.getModifiersEx()&MouseEvent.CTRL_DOWN_MASK)!=0) ) // si hay algún modificador pulsado
			return;
		//Buscamos índice del punto más cercano de la ruta correspondiente
		Point2D.Double pto=pmvt.pixel2Point(even.getX(), even.getY());
//		System.out.println(getClass().getName()+": Pulsado Boton "+even.getButton()
//				+" en posición: ("+even.getX()+","+even.getY()+")"
//				+"  ("+pto.getX()+","+pto.getY()+")  "
//		);
		int indSel=-1;
		double escala=pmvt.getEscala();
		if(indRutaActual>=0) {
			//hay ruta actual, tratamos de ver si está cerca de la ruta actual
			//indice y distancia del más cercano usando la trayectoria
			Trayectoria traActual=pmvt.getTrayectoria(indRutaActual);
			traActual.situaCoche(pto.getX(),pto.getY());
			double distMin=traActual.distanciaAlMasCercano();
//			System.out.println(getClass().getName()+": Punto más cercano a "+distMin
//					 +" escala:"+escala+ " relacion :"+(escala/distMin));
			if( ( !Double.isNaN(escala) && (distMin<=(escala/100)) )
					|| (distMin<2) ) {
				indSel=traActual.indiceMasCercano();
			}
		}
		if(indSel<0) { //pulsación lejos de la trayectoria seleccionada
			//buscamos la trayectoria más cercana
			double distTrMin=Double.MAX_VALUE;
			int indTraMin=-1;
			for(int i=0; i<tramos.size(); i++) {
				if(!pmvt.isMostrado(i)) continue; //nos lo saltamos si no está mostrado
				Trayectoria traActual=pmvt.getTrayectoria(i);
				traActual.situaCoche(pto.getX(),pto.getY());
				double distMin=traActual.distanciaAlMasCercano();
//				System.out.println(getClass().getName()+": Punto más cercano a "+distMin
//						+" indice:"+indMin+ " escala:"+escala+ " veintaba:"+escala/10);
				if( distMin<distTrMin 
						&& ( ( !Double.isNaN(escala) && (distMin<=(escala/100)) )
						|| (distMin<2) )
						) {
					indSel=traActual.indiceMasCercano();
					indTraMin=i;
					distTrMin=distMin;
				}				
			}
			if(indSel>=0) { 
				//se encontró trayectoria cercana, será la nueva seleccionada
				cambiaRutaActual(indTraMin);
			} else //no se encontró trayectoria cercana
				return;
		}
		if(even.getButton()==MouseEvent.BUTTON1)  {
			//fijamos el indice correspondiente
			per.jsDato.setValue(indSel);			
		}
		if(even.getButton()==MouseEvent.BUTTON3)  {
//			System.out.println("Mostramos el menu de punto");
			mostrarMenu(indSel, even);			
		}


	}
	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stubTODO Auto-generated method stubTODO Auto-generated method stub
		
	}

	public void mouseEntered(MouseEvent e) {
		// No hacemos nada por ahora
		
	}
	public void mouseExited(MouseEvent e) {
		//  No hacemos nada por ahora
		
	}

	private void mostrarMenu(final int ipto, MouseEvent me) {
		JPopupMenu popup = new JPopupMenu();
		JMenuItem item=null;
		
		item = new JMenuItem("Borrar Punto");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				Ruta ra=tramos.getRuta(indRutaActual);
				ra.remove(ipto);
				//para que se actualice el número de puntos
				modeloTR.fireTableCellUpdated(indRutaActual, COL_TAM);
				pmvt.setTrayectoria(indRutaActual, new Trayectoria(ra,Double.MAX_VALUE,-1));
				per.setRuta(ra, true);
				per.repaint();
				pmvt.actualiza();
			}
		});
		
		item = new JMenuItem("Borrar Hasta Principio");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				Ruta ra=tramos.getRuta(indRutaActual);
				ra.removeToBegin(ipto);
				//para que se actualice el número de puntos
				modeloTR.fireTableCellUpdated(indRutaActual, COL_TAM);
				pmvt.setTrayectoria(indRutaActual, new Trayectoria(ra,Double.MAX_VALUE,-1));
				per.setRuta(ra, true);
				per.setIndice(0); //el primero
				per.repaint();
				pmvt.actualiza();
			}
		});
		
		item = new JMenuItem("Borrar Hasta Final");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				Ruta ra=tramos.getRuta(indRutaActual);
				ra.removeToEnd(ipto);
				//para que se actualice el número de puntos
				modeloTR.fireTableCellUpdated(indRutaActual, COL_TAM);
				pmvt.setTrayectoria(indRutaActual, new Trayectoria(ra,Double.MAX_VALUE,-1));
				per.setRuta(ra, true);
				per.setIndice(ra.getNumPuntos()-1); //el último
				per.repaint();
				pmvt.actualiza();
			}
		});
		
		//Dividir ruta en dos a partir de este punto
		item = new JMenuItem("Patir desde aquí");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				Ruta ra=tramos.getRuta(indRutaActual);
				Ruta segunda=ra.divideFrom(ipto);
				//para que se actualice el número de puntos
				modeloTR.fireTableCellUpdated(indRutaActual, COL_TAM);
				pmvt.setTrayectoria(indRutaActual, new Trayectoria(ra,Double.MAX_VALUE,-1));
				
				añadeRuta(segunda, tramos.getNombre(indRutaActual)+"_B");
				//ponemos puntos y rumbo como la seleccionada
				int indNueva=tramos.size()-1;
				pmvt.setPuntos(indNueva, pmvt.isPuntos(indRutaActual));
				pmvt.setRumbo(indNueva, pmvt.isRumbo(indRutaActual));
				modeloTR.fireTableRowsInserted(indNueva, indNueva);
				per.setRuta(ra, true);
				per.setIndice(ra.getNumPuntos()-1); //el último				
				per.repaint();				
				pmvt.actualiza();
			}
		});
		
		popup.addSeparator();
		
		item = new JMenuItem("Borrar Trayectoria");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				borraTrayectoria(indRutaActual);
				modeloTR.fireTableRowsDeleted(indRutaActual, indRutaActual);
				//Cambio en las columnas de relaciones
				modeloTR.fireTableDataChanged(); //TODO optimizar para usar sólo las columnas
				pmvt.actualiza();
				indRutaActual=-1;
				per.setRuta(null); //no hay ruta seleccionada
			}
		});
		
		popup.show(me.getComponent(), me.getX(), me.getY());

	}

	/**
	 * @param row
	 */
	protected void cambiaRutaActual(int row) {
		int oldAct=indRutaActual;
		//La ruta actual siempre debe estar visible
		pmvt.setMostrado(row, true);
		pmvt.setDestacado(row, true);
		modeloTR.fireTableCellUpdated(row, COL_VISTA); //para que se muestre marca
		if(oldAct>=0) { //desmarcamos el anterior
			pmvt.setDestacado(oldAct, false);
			modeloTR.fireTableCellUpdated(oldAct, COL_ACT);
		}
		
		modeloTR.fireTableDataChanged(); //TODO cambian las columnas de sig, prio y opo
		//antes de cambiar queremos usar en íncide del más cercano
		indRutaActual=row;
		per.setRuta(tramos.getRuta(indRutaActual));
		if(pmvt.getTrayectoria(row).hayPosicionCoche())
			per.setIndice(pmvt.getTrayectoria(row).indiceMasCercano());
		pmvt.actualiza();
	}

	final static int COL_IND=0;
	final static int COL_ACT=1;
	final static int COL_VISTA=2;
	final static int COL_COLOR=3;
	final static int COL_PUNTOS=4;
	final static int COL_RUMBO=5;
	final static int COL_NOM=6;
	final static int COL_TAM=7;
	final static int COL_SIGUIENTE=8;
	final static int COL_PRIO=9;
	final static int COL_OPO=10;
	final static int COL_SUB=11;
	final static int COL_BAJ=12;
	

	class ModeloTablaRutas extends AbstractTableModel {
    	String[] nombColumnas=new String[13];
    	
    	public ModeloTablaRutas() {
			super();
			//rellenamos los Titulos
			nombColumnas[COL_IND]="I";
			nombColumnas[COL_ACT]="Act";
			nombColumnas[COL_NOM]="Nombre";
			nombColumnas[COL_TAM]="Tam";
			nombColumnas[COL_VISTA]="Ver";
			nombColumnas[COL_COLOR]="Color";
			nombColumnas[COL_PUNTOS]="Pto";
			nombColumnas[COL_RUMBO]="Rum";
			nombColumnas[COL_SIGUIENTE]="Sig.";
			nombColumnas[COL_PRIO]="Prio";
			nombColumnas[COL_OPO]="Opo";
			nombColumnas[COL_SUB]="Sub";
			nombColumnas[COL_BAJ]="Baj";
			
    	}
    			

		public int getColumnCount() {
			return nombColumnas.length;
		}

		public int getRowCount() {
			return tramos.size();
		}


		public String getColumnName(int col) {
            return nombColumnas[col];
        }
		
		public Object getValueAt(int row, int col) {
			//sacamos datos de vecRutas
			if( tramos.size()<=row)
				return null;
			switch(col) {
			case COL_IND:
				return row;
			case COL_NOM:
				return tramos.getNombre(row);
			case COL_TAM:
				return tramos.getRuta(row).getNumPuntos();
			case COL_COLOR:
				return pmvt.getColor(row);
			case COL_PUNTOS:
				return pmvt.isPuntos(row);
			case COL_RUMBO:
				return pmvt.isRumbo(row);
			case COL_VISTA:
				return pmvt.isMostrado(row);
			case COL_ACT:
				return row==indRutaActual;
			case COL_SIGUIENTE:
				if(indRutaActual<0)
					return false; //no hay trayectoria seleccionada
				else
					return tramos.isSiguiente(indRutaActual,row);
			case COL_PRIO:
				if(indRutaActual<0)
					return false; //no hay trayectoria seleccionada
				else
					return tramos.isPrioritatio(indRutaActual,row);
			case COL_OPO:
				if(indRutaActual<0)
					return false; //no hay trayectoria seleccionada
				else
					return tramos.isPrioritarioOposicion(indRutaActual,row);
			case COL_SUB:
			case COL_BAJ:
				return false; //Nunca se marcan, por eso devolvemos false
			default:
				return null;			
			}
		}
		
		public boolean isCellEditable(int row, int col) { 
        	return (col!=COL_TAM) && (col!=COL_IND);  
        }

        public void setValueAt(Object value, int row, int col) {
        	switch (col) {
        	case COL_PUNTOS:
        		pmvt.setPuntos(row, (Boolean)value);
        		break;
        	case COL_VISTA:
        		//solo desactivamos ni no es la actual
        		if((Boolean)value || row!=indRutaActual)
        			pmvt.setMostrado(row, (Boolean)value);
        		break;
        	case COL_RUMBO:
        		pmvt.setRumbo(row, (Boolean)value);
        		break;
        	case COL_ACT:
        		if(row!=indRutaActual) { //solo cambiamos al pinchar en otra fila
        			cambiaRutaActual(row);
        		}
    			break;
        	case COL_COLOR:
        		pmvt.setColor(row, (Color)value);
        		break;
        	case COL_NOM:
        		tramos.setNombre(row,(String)value);
        		break;
        	case COL_SIGUIENTE:
				if(indRutaActual<0)
					return; //no hay trayectoria seleccionada no se asigna nada
				else
					tramos.setSiguiente(indRutaActual,row,(Boolean)value);
				break;
        	case COL_PRIO:
				if(indRutaActual<0)
					return; //no hay trayectoria seleccionada no se asigna nada
				else
					tramos.setPrioritario(indRutaActual,row,(Boolean)value);
				break;
        	case COL_OPO:
				if(indRutaActual<0)
					return; //no hay trayectoria seleccionada no se asigna nada
				else
					tramos.setPrioritarioOposicion(indRutaActual,row,(Boolean)value);
				break;
        	case COL_SUB: //tenemos que subir la fila actual (si es posible)
        		if((Boolean)value && row>0) {
        			pmvt.subirTrayectoria(row);
        			tramos.subir(row);
        			modeloTR.fireTableRowsUpdated(row-1, row);
        			//el pmvt no cambia.
        		}
        		break;
        	case COL_BAJ: //tenemos que bajar la fila actual (si es posible)
        		if((Boolean)value && row<(tramos.size()-1)) {
        			pmvt.bajarTrayectoria(row);
        			tramos.bajar(row);
        			modeloTR.fireTableRowsUpdated(row, row+1);
        			//el pmvt no cambia.
        		}
        		break;
        	default:
        		return;
        	}
            fireTableCellUpdated(row, col);
            pmvt.actualiza();
        }

        public Class getColumnClass(int col) {
        	switch (col) {
        	case COL_PUNTOS:
        	case COL_RUMBO:
        	case COL_VISTA:
        	case COL_ACT:
        	case COL_SIGUIENTE:
        	case COL_PRIO:
        	case COL_OPO:
        	case COL_SUB:
        	case COL_BAJ:
        		return Boolean.class;
        	case COL_TAM:
        	case COL_IND:
        		return Integer.class;
        	case COL_COLOR:
        		return Color.class;
        	case COL_NOM:
        	default:
        		return String.class;
        	}
        }
        
        /** @return el largo maximo de los elementos que aparecen en la comulna. 
         * Se computa del maximo texto a sacar y el titulo
         */
    	public int getLargoColumna(int col) {
    		int largo=nombColumnas[col].length();
        	switch (col) {
        	case COL_NOM:
        		for(int i=0; i<tramos.size();i++)
        			largo=Math.max(largo, tramos.getNombre(i).length());
        		break;
        	case COL_TAM:
        		for(int i=0; i<tramos.size();i++)
        			largo=Math.max(largo,(" "+tramos.getRuta(i).getNumPuntos()).length());
        		break;
        	}
    		return largo*10;
    	}

    	public void actualiza() {
			fireTableDataChanged();
		}

	}
	
	/** Sacado de los ejemplo de JTable de Swing 
	 * @see http://java.sun.com/docs/books/tutorial/uiswing/examples/components/index.html#TableDialogEditDemo
	 */
	public class ColorRenderer extends JLabel implements TableCellRenderer {
		Border unselectedBorder = null;
		Border selectedBorder = null;
		boolean isBordered = true;

		public ColorRenderer(boolean isBordered) {
			this.isBordered = isBordered;
			setOpaque(true); //MUST do this for background to show up.
		}

		public Component getTableCellRendererComponent(
				JTable table, Object color,
				boolean isSelected, boolean hasFocus,
				int row, int column) {
			Color newColor = (Color)color;
			setBackground(newColor);
			if (isBordered) {
				if (isSelected) {
					if (selectedBorder == null) {
						selectedBorder = BorderFactory.createMatteBorder(2,5,2,5,
								table.getSelectionBackground());
					}
					setBorder(selectedBorder);
				} else {
					if (unselectedBorder == null) {
						unselectedBorder = BorderFactory.createMatteBorder(2,5,2,5,
								table.getBackground());
					}
					setBorder(unselectedBorder);
				}
			}
			setToolTipText("RGB value: " + newColor.getRed() + ", "
					+ newColor.getGreen() + ", "
					+ newColor.getBlue());
			return this;
		}
	}

	/** Sacado de los ejemplo de JTable de Swing 
	 * @see http://java.sun.com/docs/books/tutorial/uiswing/examples/components/index.html#TableDialogEditDemo
	 */
	public class ColorEditor extends AbstractCellEditor implements TableCellEditor,	ActionListener {
		Color currentColor;
		JButton button;
		JColorChooser colorChooser;
		JDialog dialog;
		protected static final String EDIT = "edit";

		public ColorEditor() {
			//Set up the editor (from the table's point of view),
			//which is a button.
			//This button brings up the color chooser dialog,
			//which is the editor from the user's point of view.
			button = new JButton();
			button.setActionCommand(EDIT);
			button.addActionListener(this);
			button.setBorderPainted(false);

			//Set up the dialog that the button brings up.
			colorChooser = new JColorChooser();
			dialog = JColorChooser.createDialog(button,
					"Pick a Color",
					true,  //modal
					colorChooser,
					this,  //OK button handler
					null); //no CANCEL button handler
		}

		/**
		 * Handles events from the editor button and from
		 * the dialog's OK button.
		 */
		public void actionPerformed(ActionEvent e) {
			if (EDIT.equals(e.getActionCommand())) {
				//The user has clicked the cell, so
				//bring up the dialog.
				button.setBackground(currentColor);
				colorChooser.setColor(currentColor);
				dialog.setVisible(true);

				//Make the renderer reappear.
				fireEditingStopped();

			} else { //User pressed dialog's "OK" button.
				currentColor = colorChooser.getColor();
			}
		}

		//Implement the one CellEditor method that AbstractCellEditor doesn't.
		public Object getCellEditorValue() {
			return currentColor;
		}

		//Implement the one method defined by TableCellEditor.
		public Component getTableCellEditorComponent(JTable table,
				Object value,
				boolean isSelected,
				int row,
				int column) {
			currentColor = (Color)value;
			return button;
		}
	}
	
	/** Acción para la carga de los ficheros de tramos */
	class AccionCargarTramos extends AbstractAction {

		public AccionCargarTramos() {
			super("Abrir Fichero Tramos");
		}
		
		public void actionPerformed(ActionEvent ae) {
			fc.setCurrentDirectory(new File("./Rutas/Tramos"));
			int devuelto=fc.showOpenDialog(EditaFicherosRuta.this);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=fc.getSelectedFile();
				Tramos nuevoTramos=Tramos.cargaTramos(file);
				//reemplazamos los tramos existentes
				if(nuevoTramos!=null) {
					//Borramos los tramos anteriores del panel gráfico
					for(int i=tramos.size()-1; i>=0; i--)
						pmvt.borraTrayectoria(i);
					tramos=nuevoTramos;
					//añadimos los nuevos tramos al panel gráfico
					for(int i=0; i<tramos.size();i++)
						pmvt.añadeTrayectoria(new Trayectoria(tramos.getRuta(i),Double.MAX_VALUE,-1));
					pmvt.actualiza();
					//se han cambiado todos los datos
					modeloTR.fireTableDataChanged();
					if(tramos.getNombFichDestinos()!=null)
						estableceFicheroDestino(new File(tramos.getNombFichDestinos()));
				}
			}
		}
	}

	/** Acción para salvar los ficheros de tramos */
	class AccionSalvarTramos extends AbstractAction {

		public AccionSalvarTramos() {
			super("Salvar Fichero Tramos");
		}
		
		public void actionPerformed(ActionEvent ae) {
			fc.setCurrentDirectory(new File("./Rutas/Tramos"));
			int devuelto=fc.showSaveDialog(EditaFicherosRuta.this);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=fc.getSelectedFile();
				Tramos.salvaTramos(tramos, file.getAbsolutePath());
			}
		}
	}

	/** Acción para la carga de los ficheros de rutas */
	class AccionCargarRuta extends AbstractAction {

		public AccionCargarRuta() {
			super("Abrir Fichero Ruta");
		}
		
		public void actionPerformed(ActionEvent ae) {
			fc.setCurrentDirectory(new File("./Rutas"));
			int devuelto=fc.showOpenDialog(EditaFicherosRuta.this);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=fc.getSelectedFile();
				cargaFicheroRuta(file);
			}
		}
	}

	/** Acción para la carga de los ficheros de rutas */
	class AccionEligeFicheroDestinos extends AbstractAction {

		public AccionEligeFicheroDestinos()  {
			super("Elegir Fichero Destinos");
		}
		
		public void actionPerformed(ActionEvent ae) {
			fc.setCurrentDirectory(new File("./Sitios"));
			int devuelto=fc.showOpenDialog(EditaFicherosRuta.this);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=fc.getSelectedFile();
				estableceFicheroDestino(file);
			}
		}
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		EditaFicherosRuta efr=new EditaFicherosRuta("Edita Fich. Ruta");
//		efr.cargaFichero(new File("Rutas/Universidad/Tramos_EntradaLargaSalida"));
//		efr.cargaFichero(new File("Rutas/Universidad/Tramos_NaveSalida"));
		
	}


}
