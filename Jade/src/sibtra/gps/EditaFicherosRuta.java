/**
 * 
 */
package sibtra.gps;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.AbstractAction;
import javax.swing.AbstractCellEditor;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
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
	
	public EditaFicherosRuta(String titulo) {
		
		super(titulo);
		fc=new JFileChooser(new File("./Rutas"));
//		JPanel cp=(JPanel)getContentPane();
		JPanel cp=new JPanel();
		cp.setLayout(new BoxLayout(cp,BoxLayout.PAGE_AXIS));
//		cp.setBorder(BorderFactory.createMatteBorder(3, 3, 3, 3, Color.YELLOW));
		
		
		pmvt=new PanelMuestraVariasTrayectorias();
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
	 * @param file Fichero a cargar
	 */
	protected void cargaFichero(File file) {
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

	public void itemStateChanged(ItemEvent e) {
	}
	
	/** Para gestionar los cambios en el punto seleccionado en el {@link #per} */
	public void stateChanged(ChangeEvent arg0) {
		GPSData ultPto=per.ruta.getPunto((Integer)per.jsDato.getValue());
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
	 * Si es 3 mostramos menu del punto
	 * @param even evento
	 */
	public void mousePressed(MouseEvent even) {
		if(indRutaActual<0)
			return; //no hay ruta
		//Buscamos índice del punto más cercano de la ruta correspondiente
		Point2D.Double pto=pmvt.pixel2Point(even.getX(), even.getY());
		System.out.println(getClass().getName()+": Pulsado Boton "+even.getButton()
				+" en posición: ("+even.getX()+","+even.getY()+")"
				+"  ("+pto.getX()+","+pto.getY()+")  "
		);
		//indice y distancia del más cercano usando la trayectoria
		Trayectoria traActual=pmvt.getTrayectoria(indRutaActual);
		traActual.situaCoche(pto.getX(),pto.getY());
		double distMin=traActual.distanciaAlMasCercano();
		int indMin=traActual.indiceMasCercano();
		double escala=pmvt.getEscala();
		System.out.println(getClass().getName()+": Punto más cercano a "+distMin
				+" indice:"+indMin+ " escala:"+escala+ " veintaba:"+escala/10);
		if(!Double.isNaN(escala)) {
			if(distMin>(escala/10))
				return; //pulsación muy lejana
		} else if (distMin>2)
			return; //si fuera de escala la distancia debe ser <2

		if(even.getButton()==MouseEvent.BUTTON1)  {
			//fijamos el indice correspondiente
			per.jsDato.setValue(indMin);			
		}
		if(even.getButton()==MouseEvent.BUTTON3)  {
			System.out.println("Mostramos el menu de punto");
			mostrarMenu(indMin, even);			
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
				pmvt.actualiza();
				per.setRuta(ra, true);
				per.repaint();
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
				pmvt.actualiza();
				per.setRuta(ra, true);
				per.repaint();
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
				pmvt.actualiza();
				per.setRuta(ra, true);
				per.repaint();
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
				
				pmvt.actualiza();
				per.setRuta(ra, true);
				per.repaint();
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
		indRutaActual=row;
		per.setRuta(tramos.getRuta(indRutaActual));
	}

	final static int COL_ACT=0;
	final static int COL_VISTA=1;
	final static int COL_COLOR=2;
	final static int COL_PUNTOS=3;
	final static int COL_RUMBO=4;
	final static int COL_NOM=5;
	final static int COL_TAM=6;
	final static int COL_SIGUIENTE=7;
	final static int COL_PRIO=8;
	final static int COL_OPO=9;
	

	class ModeloTablaRutas extends AbstractTableModel {
    	String[] nombColumnas=new String[10];
    	
    	public ModeloTablaRutas() {
			super();
			//rellenamos los Titulos
			nombColumnas[COL_ACT]="Act";
			nombColumnas[COL_NOM]="Nombre";
			nombColumnas[COL_TAM]="Tamaño";
			nombColumnas[COL_VISTA]="Ver";
			nombColumnas[COL_COLOR]="Color";
			nombColumnas[COL_PUNTOS]="Puntos";
			nombColumnas[COL_RUMBO]="Rumbo";
			nombColumnas[COL_SIGUIENTE]="Sig.";
			nombColumnas[COL_PRIO]="Prio";
			nombColumnas[COL_OPO]="Opo";
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
			default:
				return null;			
			}
		}
		
		public boolean isCellEditable(int row, int col) { 
        	return (col!=COL_TAM);  
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
        			int oldAct=indRutaActual;
        			//La ruta actual siempre debe estar visible
        			pmvt.setMostrado(row, true);
        			fireTableCellUpdated(row, COL_VISTA); //para que se muestre marca
        			if(oldAct>=0) //desmarcamos el anterior
        				fireTableCellUpdated(oldAct, COL_ACT);
        			cambiaRutaActual(row);
        			fireTableDataChanged(); //TODO cambian las columnas de sig, prio y opo
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
        		return Boolean.class;
        	case COL_TAM:
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
			int devuelto=fc.showOpenDialog(EditaFicherosRuta.this);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=fc.getSelectedFile();
				cargaFichero(file);
			}
		}
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		EditaFicherosRuta efr=new EditaFicherosRuta("Examina Fich Ruta");
//		efr.cargaFichero(new File("Rutas/Universidad/Tramos_EntradaLargaSalida"));
//		efr.cargaFichero(new File("Rutas/Universidad/Tramos_NaveSalida"));
		
	}


}
