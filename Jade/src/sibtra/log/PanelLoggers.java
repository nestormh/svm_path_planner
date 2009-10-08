package sibtra.log;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.MouseEvent;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.table.AbstractTableModel;

import sibtra.util.SalvaMATv4;

/**
 * Panel que permitirá gestionar los loggers:
 * seleccionar los loggers a activar, activarlos, salvar datos a fichero, etc.
 * @author alberto
 *
 */
public class PanelLoggers extends JTabbedPane {

	private Action accionRefrescar;
	private Action accionLimpiar;
	private Action accionDesactivar;
	private Action accionActivar;
	private Action accionSalvar;

	private SpinnerNumberModel segActiva;
	private JTable tablaLoggers;

	/** Clase para apuntar en un vector el logger y si está seleccionado */
	class ApuntaLog {
		Logger la=null;
		boolean sel=false; //si esta seleccionado
		public ApuntaLog(Logger la,boolean sel) {
			this.la=la;
			this.sel=sel;
		}
	}
	/** Vector con los loger considerados y si están seleccionados */
	Vector<ApuntaLog> vecLA=null;
	private ModeloTablaLoggers modeloTL;

	final static int COL_SEL=0;
	final static int COL_NOM=1;
	final static int COL_CLASS=2;
	final static int COL_ACTIVA=3;
	final static int COL_SIZE=4;
	final static int COL_CAPA=5;

	public PanelLoggers() {
		super(JTabbedPane.TOP, JTabbedPane.WRAP_TAB_LAYOUT);
		
		{ //primera pestaña para activar
			JPanel panAct=new JPanel();
			panAct.setLayout(new BoxLayout(panAct,BoxLayout.PAGE_AXIS));
			//arriba el titulo
			JLabel titulo=new JLabel("Loggers Definidos:");
			titulo.setAlignmentX(Component.LEFT_ALIGNMENT);
			panAct.add(titulo);
			

			panAct.add(Box.createRigidArea(new Dimension(0,5)));
			
			//Donde seleccionar los loggers
			modeloTL=new ModeloTablaLoggers();
			tablaLoggers=new JTable(modeloTL) {
				/** Saca descripcion en columna del nombre */
		        public String getToolTipText(MouseEvent e) {
		            String tip = null;
		            java.awt.Point p = e.getPoint();
		            int rowIndex = rowAtPoint(p);
		            int colIndex = columnAtPoint(p);
		            int realColumnIndex = convertColumnIndexToModel(colIndex);

		            if (realColumnIndex == COL_NOM) { //Columna del nombre
		                tip = modeloTL.getDescripcion(rowIndex);
		            } else { //another column
		                tip = super.getToolTipText(e);
		            }
		            return tip;
		        }				
			};
	        tablaLoggers.setPreferredScrollableViewportSize(new Dimension(500, 70));
//	        table.setAutoCreateRowSorter(true);
	        //fijamos tamaños preferidos
	        for(int i=0; i<modeloTL.getColumnCount();i++)
	        	tablaLoggers.getColumnModel().getColumn(i).setPreferredWidth(
	        			modeloTL.getLargoColumna(i)	        			
	        	);

			panAct.add(new JScrollPane(tablaLoggers));

			panAct.add(Box.createRigidArea(new Dimension(0,5)));

			{ //Botones activacion, limpiado y salvado
				JPanel buttonPane = new JPanel();
				buttonPane.setMaximumSize(new Dimension(1200,60));
				buttonPane.setMinimumSize(new Dimension(700,60));
				buttonPane.setLayout(new BoxLayout(buttonPane, BoxLayout.LINE_AXIS));
				buttonPane.setBorder(BorderFactory.createEmptyBorder(0, 10, 10, 10));

				buttonPane.add(Box.createHorizontalGlue());
				
				accionRefrescar=new AbstractAction("Refrescar") {
//					{ setName("Refrescar"); }
					public void actionPerformed(ActionEvent e) {
						repinta();
					}
				};
				buttonPane.add(new JButton(accionRefrescar));

				buttonPane.add(Box.createRigidArea(new Dimension(10, 0)));
				
				accionLimpiar=new AbstractAction("Limpiar") {
					public void actionPerformed(ActionEvent e){
						//actuamos sobre aquellos loggers seleccionados
						for(int i=0; i<vecLA.size(); i++){
							if(!vecLA.get(i).sel) continue; // nos saltamso los no seleccionados
							Logger la=vecLA.get(i).la;
							la.clear();
//							System.out.println("Limpiando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
//									+" para "+sgAct+" segundos.");
						}
						tablaLoggers.repaint();
					}
				};
				buttonPane.add(new JButton(accionLimpiar));
				
				buttonPane.add(Box.createRigidArea(new Dimension(10, 0)));
				
				accionDesactivar=new AbstractAction("Desactiva") {
					public void actionPerformed(ActionEvent e){
						//actuamos sobre aquellos loggers seleccionados
						for(int i=0; i<vecLA.size(); i++){
							if(!vecLA.get(i).sel) continue; // nos saltamso los no seleccionados
							Logger la=vecLA.get(i).la;
							la.desactiva();
//							System.out.println("Desactivando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
//									+" para "+sgAct+" segundos.");
						}
						tablaLoggers.repaint();
					}
				};
				buttonPane.add(new JButton(accionDesactivar));
				
				buttonPane.add(Box.createRigidArea(new Dimension(10, 0)));
				
				accionActivar=new AbstractAction("Activa") {
					public void actionPerformed(ActionEvent e){
						//actuamos sobre aquellos loggers seleccionados
						int sgAct=segActiva.getNumber().intValue();
						for(int i=0; i<vecLA.size(); i++){
							if(!vecLA.get(i).sel) continue; // nos saltamso los no seleccionados
							Logger la=vecLA.get(i).la;
							la.activa(sgAct);
//							System.out.println("Activando Logger:"+la.objeto.getClass().getName()+":"+la.nombre
//									+" para "+sgAct+" segundos.");
						}
						tablaLoggers.repaint();
					}
				};
				buttonPane.add(new JButton(accionActivar));
				
				buttonPane.add(new JLabel(" para "));
				segActiva=new SpinnerNumberModel(300,10,3600,5);
				buttonPane.add(new JSpinner(segActiva));
				buttonPane.add(new JLabel(" sg."));
				
				
				accionSalvar=new AbstractAction("Salvar") {
					public void actionPerformed(ActionEvent ae){
						String nombBase="Datos/PruPanLog";
						String nombCompleto=nombBase+new SimpleDateFormat("yyyyMMddHHmm").format(new Date())
						+".mat"
						;
						System.out.println("Escribiendo en Fichero "+nombCompleto);
						try {
							SalvaMATv4 smv4=new SalvaMATv4(nombCompleto);
							for(int i=0; i<vecLA.size(); i++){
								if(!vecLA.get(i).sel) continue; // nos saltamos los no seleccionados
								Logger la=vecLA.get(i).la;
								la.vuelcaMATv4(smv4);
								System.out.println("Volcando Logger:"+la.objeto.getClass().getName()+":"+la.nombre);
							}		
							smv4.close();
						} catch (IOException e) {
							// TODO Bloque catch generado automáticamente
							e.printStackTrace();
						}
					}
				};
				buttonPane.add(new JButton(accionSalvar));
				panAct.add(buttonPane);
			}
			//Añadimos panel a la pestaña
			addTab("Activacion", panAct);			
		}
	}

	/** Modelo de nuestra tabla de loggers */
    class ModeloTablaLoggers extends AbstractTableModel {
    	
    	String[] nombColumnas=new String[6];
		private static final String etiquetaActivo="ACTIVO";
    	
    	
    	public ModeloTablaLoggers() {
    		super();
    		//rellenamos array de titulos
        	nombColumnas[COL_SEL]="Sel.";
        	nombColumnas[COL_NOM]="Logger"; 
        	nombColumnas[COL_CLASS]="Clase"; 
        	nombColumnas[COL_ACTIVA]="Activado"; 
        	nombColumnas[COL_SIZE]="Tamaño"; 
        	nombColumnas[COL_CAPA]="Capacidad";
        	
    		//Apuntamos los logger disponibles
        	int NumLoggers=(LoggerFactory.vecLoggers!=null)?LoggerFactory.vecLoggers.size():0;
    		vecLA=new Vector<ApuntaLog>(NumLoggers);
    		for(Iterator<Logger> lit=LoggerFactory.vecLoggers.iterator(); lit.hasNext();) {
    			Logger la=lit.next();
    			vecLA.add(new ApuntaLog(la,true));
    		}
    	}
    	
		public String getColumnName(int col) {
            return nombColumnas[col].toString();
        }
        public int getRowCount() { return LoggerFactory.vecLoggers.size(); }
        public int getColumnCount() { return nombColumnas.length; }
        public Object getValueAt(int row, int col) {
        	// Sacamos los datos de vecLA
        	if(vecLA==null || vecLA.size()<=row)
        		return null;
        	Logger la=vecLA.get(row).la;
        	switch (col) {
        	case COL_SEL:
        		return vecLA.get(row).sel;
        	case COL_NOM:
        		return la.nombre;
        	case COL_CLASS:
        		return la.getNombreClase();
        	case COL_ACTIVA:
        		return (la.isActivo()?etiquetaActivo:"    ");
        	case COL_SIZE:
        		return (la.tiempos!=null)?la.tiempos.size():0;
        	case COL_CAPA:
        		return (la.tiempos!=null)?la.tiempos.capacity():0;
        	default:
        		return null;
        	}
        }
        public void setValueAt(Object value, int row, int col) {
        	if(col!=0) return;
            vecLA.get(row).sel = (Boolean)value;
            fireTableCellUpdated(row, col);
        }

        public boolean isCellEditable(int row, int col) { 
        	return (col==0); //solo la columna 0 es editable 
        }
        public Class getColumnClass(int col) {
        	switch (col) {
        	case COL_SEL:
        		return Boolean.class;
        	case COL_NOM:
        	case COL_CLASS:
        	case COL_ACTIVA:
        		return String.class;
        	case COL_SIZE:
        	case COL_CAPA:
        		return Integer.class;
        	default:
        		return String.class;
        	}
        }

        /** Revisa la lista de loggers para ver los que se han añadido o borrado 
         * */
        void actualizaListaLoggers() {
        	if(LoggerFactory.vecLoggers==null || LoggerFactory.vecLoggers.size()==0)
        		return;
        	boolean modificada=false; //apuntamos si hay cambio para actualizar tabla
        	//Buscamos nuevos loggers añadidos
        	for(int i=0; i<LoggerFactory.vecLoggers.size(); i++) {
        		Logger la=LoggerFactory.vecLoggers.get(i);
        		boolean encontrado=false;
        		for(int j=0; !encontrado && j<vecLA.size(); j++)
        			encontrado=(la==vecLA.get(j).la);
        		if(encontrado) continue;
        		//si no se encontró lo añadimos
        		vecLA.add(new ApuntaLog(la,true));
        		modificada=true;
        	}
        	//eliminamos loggers borrados
        	for(int i=0; i<vecLA.size(); i++) {
        		ApuntaLog alAct=vecLA.get(i);
        		boolean encontrado=false;
        		for(int j=0; !encontrado && j<LoggerFactory.vecLoggers.size(); j++)
        			encontrado=(alAct.la==LoggerFactory.vecLoggers.get(j));
        		if(encontrado) continue;
        		//si no se encontró lo borramos
        		vecLA.removeElementAt(i);
        		i--; //porque se ha desplazado el vector
        		modificada=true;
        	}
        	// Indicamos datos han cambiado
        	if (modificada)
        		fireTableDataChanged();
        }
        /** @return el largo maximo de los elementos que aparecen en la comulna. 
         * Se computa del maximo texto a sacar y el titulo
         */
    	public int getLargoColumna(int col) {
    		int largo=nombColumnas[col].length();
    		int lact=0;
        	// Sacamos los datos de vecLA
        	if(vecLA==null)
        		return 0;
        	switch (col) {
        	case COL_SEL:
        		break;
        	case COL_NOM:
        		for(int i=0; i<vecLA.size();i++)
        			if((lact=vecLA.get(i).la.nombre.length())>largo)
        				largo=lact;
        		break;
        	case COL_CLASS:
        		for(int i=0; i<vecLA.size();i++)
        			if((lact=vecLA.get(i).la.getNombreClase().length())>largo)
        				largo=lact;
        		break;
        	case COL_ACTIVA:
        		largo=Math.max(largo,etiquetaActivo.length()+6); //largo de la etiqueta ACTIVO
        		break;
        	case COL_SIZE:
        	case COL_CAPA:
        		largo=Math.max(largo,(" "+Integer.MAX_VALUE).length());
        		break;
        	}
    		
    		return largo*10;
    	}

        public String getDescripcion(int row) {
			return vecLA.get(row).la.descripcion;
		}

    }

	/** programamos la actualizacion del panel */
	public void repinta() {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				((ModeloTablaLoggers)(tablaLoggers.getModel())).actualizaListaLoggers();
				repaint();
			}
		});
	}
	
	public static void main(String[] args) {
		Object este=new Object();
		
		JFrame ventana=new JFrame("PanelLoggers");
		ventana.add(new PanelLoggers());
		
		ventana.pack();
		ventana.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		ventana.setVisible(true);
		
		
		try { Thread.sleep(4000); } catch (Exception e){}

		Logger l1=LoggerFactory.nuevoLoggerArrayDoubles(este, "l1");
		System.out.println("Creamos primer logger");
		try { Thread.sleep(5000); } catch (Exception e){}

		Logger l2=LoggerFactory.nuevoLoggerArrayDoubles(este, "l2");
		System.out.println("Creamos segundo logger");
		try { Thread.sleep(5000); } catch (Exception e){}

		LoggerFactory.borraLogger(l1);
		System.out.println("Borramos primer logger");
		try { Thread.sleep(5000); } catch (Exception e){}
		
		
	}

	/**
	 * @return el accionActivar
	 */
	public Action getAccionActivar() {
		return accionActivar;
	}

	/**
	 * @return el accionDesactivar
	 */
	public Action getAccionDesactivar() {
		return accionDesactivar;
	}

	/**
	 * @return el accionLimpiar
	 */
	public Action getAccionLimpiar() {
		return accionLimpiar;
	}

	/**
	 * @return el accionRefrescar
	 */
	public Action getAccionRefrescar() {
		return accionRefrescar;
	}

	/**
	 * @return el accionSalvar
	 */
	public Action getAccionSalvar() {
		return accionSalvar;
	}
		
}
