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
import java.util.Vector;

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

	class DatosRuta {
		Ruta rt;
		String nombre;
		int indice;
		//Resto de detalles lo sacamos del  PanelMuestraVariasTrayectorias
		public DatosRuta(Ruta ruta, String nom, int ind) {
			rt=ruta;
			nombre=nom;
			indice=ind;
		}
	}
	
	protected Vector<DatosRuta> vecDRutas=new Vector<DatosRuta>();
//	private Ruta rutaEspacial=null, rutaTemporal=null, rutaActual=null;
//	private Trayectoria traActual=null;

	protected int indRutaActual=-1; //ninguna seleccionada
	
	private PanelExaminaRuta per;
	
	/** Barra de menu ventana principal */
	JMenuBar barraMenu;
	/** Menu de archivo dentro de la ventana principal */
	JMenu menuArchivo;
	/** Boton de salir del menu de archivo */
	private JMenuItem miSalir;
	/** Boton de abrir ruta de archivo */
	private JMenuItem miAbrir;

	private JFileChooser fc;
	private PanelMuestraVariasTrayectorias pmvt;

	private ModeloTablaRutas modeloTR;
	
	public EditaFicherosRuta(String titulo) {
		
		super(titulo);
		fc=new JFileChooser(new File("./Rutas"));
//		JPanel cp=(JPanel)getContentPane();
		JPanel cp=new JPanel();
		cp.setLayout(new BoxLayout(cp,BoxLayout.PAGE_AXIS));
		cp.setBorder(BorderFactory.createMatteBorder(3, 3, 3, 3, Color.YELLOW));
		
		
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
		JTable tablaRutas=new JTable(modeloTR);
		tablaRutas.setPreferredScrollableViewportSize(new Dimension(500, 70));
		tablaRutas.setColumnSelectionAllowed(false);
		tablaRutas.setRowSelectionAllowed(true);
		tablaRutas.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		//para representación y edición del color
		tablaRutas.setDefaultRenderer(Color.class, new ColorRenderer(true));
		tablaRutas.setDefaultEditor(Color.class, new ColorEditor());

        //fijamos tamaños preferidos
        for(int i=0; i<modeloTR.getColumnCount();i++)
        	tablaRutas.getColumnModel().getColumn(i).setPreferredWidth(
        			modeloTR.getLargoColumna(i)	        			
        	);

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
        splitPanel.setBorder(BorderFactory.createMatteBorder(3, 3, 3, 3, Color.GREEN));
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

        miAbrir=new JMenuItem("Abrir Ruta");
        miAbrir.setAccelerator(KeyStroke.getKeyStroke( KeyEvent.VK_O,KeyEvent.CTRL_MASK));
        miAbrir.addActionListener(this);
        menuArchivo.add(miAbrir);

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


	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==miSalir) {
			Terminar();
			return;
		}
//		if(e.getSource()==jmiAcercaDe) {
//		JOptionPane.showMessageDialog(ventanaPrincipal, "VERDINO: ISAATC ULL");
//		}
		if(e.getSource()==miAbrir) {
			int devuelto=fc.showOpenDialog(this);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=fc.getSelectedFile();
				cargaFichero(file);
			}
			return;
		}
	}


	/**
	 * @param file Fichero a cargar
	 */
	protected void cargaFichero(File file) {
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			Ruta ruta=(Ruta)ois.readObject();
			vecDRutas.add(new DatosRuta(ruta,file.getName()+"_Espacial"
					,pmvt.añadeTrayectoria(new Trayectoria(ruta) ) )
			);

			ruta=(Ruta)ois.readObject();
			vecDRutas.add(new DatosRuta(ruta,file.getName()+"_Temporal"
					,pmvt.añadeTrayectoria(new Trayectoria(ruta) ) )
			);
			ois.close();
			
		} catch (IOException ioe) {
			System.err.println("Error al abrir el fichero " + file.getName());
			System.err.println(ioe.getMessage());
		} catch (ClassNotFoundException cnfe) {
			System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
		}
		modeloTR.actualiza();
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
		Trayectoria traActual=pmvt.getTrayectoria(vecDRutas.get(indRutaActual).indice);
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
		JMenuItem item = new JMenuItem("Borrar Punto");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				DatosRuta dtRtAct=vecDRutas.get(indRutaActual);
				dtRtAct.rt.remove(ipto);
				pmvt.setTrayectoria(dtRtAct.indice, new Trayectoria(dtRtAct.rt));
				pmvt.actualiza();
			}
		});
		item = new JMenuItem("Borrar Hasta Principio");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				DatosRuta dtRtAct=vecDRutas.get(indRutaActual);
				dtRtAct.rt.removeToBegin(ipto);
				pmvt.setTrayectoria(dtRtAct.indice, new Trayectoria(dtRtAct.rt));
				pmvt.actualiza();
			}
		});
		item = new JMenuItem("Borrar Hasta Final");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				DatosRuta dtRtAct=vecDRutas.get(indRutaActual);
				dtRtAct.rt.removeToEnd(ipto);
				pmvt.setTrayectoria(dtRtAct.indice, new Trayectoria(dtRtAct.rt));
				pmvt.actualiza();
			}
		});
		popup.show(me.getComponent(), me.getX(), me.getY());

	}

	/**
	 * @param row
	 */
	protected void cambiaRutaActual(int row) {
		indRutaActual=row;
		per.setRuta(vecDRutas.get(indRutaActual).rt);
	}

	final static int COL_ACT=0;
	final static int COL_NOM=1;
	final static int COL_TAM=2;
	final static int COL_VISTA=3;
	final static int COL_COLOR=4;
	final static int COL_PUNTOS=5;
	final static int COL_RUMBO=6;

	class ModeloTablaRutas extends AbstractTableModel {
    	String[] nombColumnas=new String[7];
    	
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
    	}
    			

		public int getColumnCount() {
			return nombColumnas.length;
		}

		public int getRowCount() {
			return vecDRutas.size();
		}


		public String getColumnName(int col) {
            return nombColumnas[col];
        }
		public Object getValueAt(int row, int col) {
			//sacamos datos de vecRutas
			if(vecDRutas==null || vecDRutas.size()<=row)
				return null;
			DatosRuta dra=vecDRutas.get(row);
			switch(col) {
			case COL_NOM:
				return dra.nombre;
			case COL_TAM:
				return dra.rt.getNumPuntos();
			case COL_COLOR:
				return pmvt.getColor(dra.indice);
			case COL_PUNTOS:
				return pmvt.isPuntos(dra.indice);
			case COL_RUMBO:
				return pmvt.isRumbo(dra.indice);
			case COL_VISTA:
				return pmvt.isMostrado(dra.indice);
			case COL_ACT:
				return row==indRutaActual;
			default:
				return null;			
			}
		}
		
		public boolean isCellEditable(int row, int col) { 
        	return (col==COL_VISTA) || (col==COL_PUNTOS)
        	|| (col==COL_RUMBO) || (col==COL_ACT) || (col==COL_COLOR);  
        }

        public void setValueAt(Object value, int row, int col) {
			DatosRuta dra=vecDRutas.get(row);
        	switch (col) {
        	case COL_PUNTOS:
        		pmvt.setPuntos(dra.indice, (Boolean)value);
        		break;
        	case COL_VISTA:
        		//solo desactivamos ni no es la actual
        		if((Boolean)value || row!=indRutaActual)
        			pmvt.setMostrado(dra.indice, (Boolean)value);
        		break;
        	case COL_RUMBO:
        		pmvt.setRumbo(dra.indice, (Boolean)value);
        		break;
        	case COL_ACT:
        		if(row!=indRutaActual) { //solo cambiamos al pinchar en otra fila
        			int oldAct=indRutaActual;
        			//La ruta actual siempre debe estar visible
        			pmvt.setMostrado(dra.indice, true);
        			fireTableCellUpdated(row, COL_VISTA); //para que se muestre marca
        			if(oldAct>=0) //desmarcamos el anterior
        				fireTableCellUpdated(oldAct, COL_ACT);
        			cambiaRutaActual(row);
        			break;
        		}
        	case COL_COLOR:
        		pmvt.setColor(dra.indice, (Color)value);
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
    		int lact=0;
        	// Sacamos los datos de vecLA
        	if(vecDRutas==null)
        		return 0;
        	switch (col) {
        	case COL_PUNTOS:
        	case COL_RUMBO:
        	case COL_VISTA:
        		break;
        	case COL_NOM:
        		for(int i=0; i<vecDRutas.size();i++)
        			if((lact=vecDRutas.get(i).nombre.length())>largo)
        				largo=lact;
        		break;
        	case COL_COLOR:
        		for(int i=0; i<vecDRutas.size();i++)
        			if((lact=pmvt.getColor(vecDRutas.get(i).indice).toString().length())>largo)
        				largo=lact;
        		break;
        	case COL_TAM:
        		largo=Math.max(largo,(" "+Integer.MAX_VALUE).length());
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
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		EditaFicherosRuta efr=new EditaFicherosRuta("Examina Fich Ruta");
		efr.cargaFichero(new File("Rutas/Universidad/Tramos_EntradaLargaSalida"));
		efr.cargaFichero(new File("Rutas/Universidad/Tramos_NaveSalida"));
		
	}


}
