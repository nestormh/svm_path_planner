/**
 * 
 */
package sibtra.agentes;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.HeadlessException;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.StringTokenizer;
import java.util.regex.Pattern;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.gps.GPSData;
import sibtra.lms.BarridoAngular;

import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.domain.DFService;
import jade.domain.FIPAException;
import jade.domain.FIPANames;
import jade.domain.FIPAAgentManagement.DFAgentDescription;
import jade.domain.FIPAAgentManagement.ServiceDescription;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.UnreadableException;
import jade.proto.SimpleAchieveREInitiator;

/**
 * Este agente pedirá las distancias al AgenteLMS y las presentara en una ventana 
 * 
 * @author alberto
 *
 */
public class AgenteMuestraPunto extends Agent {
	
	/**
	 * Ventana que contendrá el PanelMuestraBarrido
	 */
	protected agenteMPGUI miGui=null;
	
	/** Lista de agentes encontrados */
	protected AID[] agentesGPS=null;

	protected ComportaEscaneoContinuo compEscaneoContinuo;
	
	protected ComportaPidePto compPideBarrido;

	private boolean logs;
	
	/**
	 * Metodo para emitir los mensajes de Log
	 */
	protected void agentLog(String m) {
		if(logs)
			System.out.println(getAID().getName()+
					": "+System.currentTimeMillis()+": "+m);
	}

	/**
	 * Inicialización:
	 * <ol>
	 *   <li>Creamos ventana gráfica y la mostramos
	 *   <li>Se registra en el DF
	 *   <li>Espara un rato
	 *   <li>Arranca en comportamiento {@link ComportaEscaneoContinuo}
	 * </ol> 
	 *  
	 * @see jade.core.Agent#setup()
	 */
	@Override
	protected void setup() {
		//super.setup();
		agentLog("Arranca");
		
		logs=true;
		//analizamos argumentos
		Object[] args = getArguments();
		if (args!=null) 
			for(int i=0; i<args.length; i++) {
				String aa=args[i].toString();
				agentLog("Argumento "+ i + " es '"+aa+"'");
				if(aa.startsWith("logs:") && aa.length()>"logs:".length() ) {
					logs=aa.substring(aa.indexOf(":")+1).equalsIgnoreCase("on");
				}
			}

		
		//Nos registramos en el DF
		DFAgentDescription  dfd = new DFAgentDescription();
		dfd.setName(getAID());
		ServiceDescription  sd = new ServiceDescription();
		sd.setType("muestra-pto");
		sd.setName("mostrador-pto");
		dfd.addServices(sd);
		try {
			DFService.register(this, dfd);
		}
		catch (FIPAException fe) {
			fe.printStackTrace();
		}

		//Creamos todos los comportamientos
		compPideBarrido=new ComportaPidePto();
		compEscaneoContinuo=new ComportaEscaneoContinuo();
		
		//Creamos la ventana grafica y la mostramos
		miGui=new agenteMPGUI(getLocalName());

		//Por defecto estamos en modo no continuo.
		//el GUI añadirá los comportamientos.
	}

	/**
	 * Teminación del agente.
	 * Se desregistra y cierra ventana si existe.
	 * @see jade.core.Agent#takeDown()
	 */
	@Override
	protected void takeDown() {
		// Deregister from the yellow pages
		try {
			DFService.deregister(this);
		}
		catch (FIPAException fe) {
			fe.printStackTrace();
		}

		// Close the GUI
		if(miGui != null ){
			agentLog("Cerramos GUI");
			miGui.dispose();
		}
		// Printout a dismissal message
		agentLog("terminado.");

	}
	
	/**
	 * Devuelve el primer agente GPS de la lista. 
	 * Si no hay ninguno trata de consegirlo.
	 * @return null si no hay, caso contrario el primer agente LMS
	 */
	protected AID agenteGPS() {
		// Obetenemos la lista de agentes generadores si no hay aún
		if (agentesGPS==null || agentesGPS.length==0) {
			DFAgentDescription template = new DFAgentDescription();
			ServiceDescription sd = new ServiceDescription();
			sd.setType("comunica-GPS");
			template.addServices(sd);
			try {
				DFAgentDescription[] result = DFService.search(AgenteMuestraPunto.this, template);
				String s=new String("Encontramos los siguientes agentes GPS:");
				agentesGPS = new AID[result.length];	
				for (int i = 0; i < result.length; ++i) {
					agentesGPS[i] = result[i].getName();
					s+="\n\t"+agentesGPS[i].getName();
				}
				agentLog(s);
			}
			catch (FIPAException fe) {
				fe.printStackTrace();
			}
		}
		if(agentesGPS!=null && agentesGPS.length>=1)
			return agentesGPS[0];
		else
			return null;
	}
	
	//Clase privada con el GUI
	/**
	 * Muestra la ventana con el panel para mostrar los datos del punto GPS y demás controles
	 */
	private class agenteMPGUI extends JFrame implements ActionListener, ChangeListener {


		/**
		 * Etiqueta donde se inica la configuración obtenida
		 */
		JLabel labelConfiguración;

		/**
		 * Modelo del spiner de frecuencia de barrido continuo
		 */
		private SpinnerNumberModel modelFrecEscaneo;

		/**
		 * Bonton para pedir barridos manualmente
		 */
		private JButton jbPidePunto;

		/**
		 * Boton donde se marca si se está haciendo escaneo continuo
		 */
		private JCheckBox cbEscaneo;

		/**
		 * Etiqueta de "Esperando" que se activará cuando se espera un barrido
		 */
		private JLabel jlEsperando;

		private JLabel jlLat;

		private JLabel jlLon;

		private JLabel jlAltura;
		
		private static final String comandoBotonPideBarrido="PideDistancia";
		private static final String comandoBotonEscaneoContinuo="EscaneoContinuo";
		
		/**
		 * Añadimos los elementos que queremos que aparezcan.
		 * @param title Titulo
		 * @throws HeadlessException
		 */
		public agenteMPGUI(String title) throws HeadlessException {
			super(title);

			addWindowListener(new   WindowAdapter() {
				public void windowClosing(WindowEvent e) {
					AgenteMuestraPunto.this.doDelete();
				}
			} );
			
			//Ponemos las Etiquetas de las distancias por zonas
			{
				Dimension sepH=new Dimension(15,0);
				//Etiquetas con nombres de zonas y su distancia
				JPanel jpD=new JPanel();
				jpD.setLayout(new BoxLayout(jpD,BoxLayout.LINE_AXIS));
				jpD.setBorder(
						BorderFactory.createCompoundBorder(
								BorderFactory.createEmptyBorder(5, 5, 5, 5)
								,BorderFactory.createLineBorder(Color.BLACK)
						)
				);
				
				jpD.add(Box.createHorizontalStrut(15));
				
				jpD.add(new JLabel("Longitud: "));    


				jpD.add(Box.createHorizontalStrut(15));
				jlLat=new JLabel("00.000");
				//jlZa.setForeground(new Color(43,105,3));  // or Color.RED
			    Font Grande = jlLat.getFont().deriveFont(20.0f);
			    jlLat.setFont(Grande);
				jlLat.setHorizontalAlignment(JLabel.CENTER);
				jlLat.setEnabled(false);
				jpD.add(jlLat);				
				jpD.add(Box.createHorizontalStrut(15));
				
				jpD.add(new JLabel("Longitud: "));
				jpD.add(Box.createHorizontalStrut(15));
				jlLon=new JLabel("00.000");
				jlLon.setHorizontalAlignment(JLabel.CENTER);
				jlLon.setEnabled(false);
				jlLon.setFont(Grande);
				jpD.add(jlLon);				
				jpD.add(Box.createHorizontalStrut(15));
				
				jpD.add(new JLabel("Zona C: "));
				jpD.add(Box.createHorizontalStrut(15));
				jlAltura=new JLabel("00.000");
				jlAltura.setHorizontalAlignment(JLabel.CENTER);
				jlAltura.setEnabled(false);
				jlAltura.setFont(Grande);
				jpD.add(jlAltura);				
				jpD.add(Box.createHorizontalStrut(15));
				
				add(jpD,BorderLayout.CENTER);
			}


			//ponemos las dos líneas (spiners y configuración) en un box layout
			JPanel jpSur=new JPanel();
			jpSur.setLayout(new BoxLayout(jpSur,BoxLayout.PAGE_AXIS));
			
			{  //Siguiente linea con lo relativo a cundo se realiza escaneo

				JPanel jpCuandoBarrido=new JPanel(new FlowLayout(FlowLayout.CENTER,10,4));
				//Check Box selección escaneo continulo
				cbEscaneo=new JCheckBox("Escaneo Continuo");
				cbEscaneo.setActionCommand(comandoBotonEscaneoContinuo);
				cbEscaneo.addActionListener(this);
				jpCuandoBarrido.add(cbEscaneo);

				//Spiner frecuencia escaneo continuo
				modelFrecEscaneo=new SpinnerNumberModel(1000,50,60000,5);
				modelFrecEscaneo.addChangeListener(this);
				JPanel jpa=new JPanel();
				jpa.add(new JLabel("Mili esc"));
				jpa.add(new JSpinner(modelFrecEscaneo));
				jpa.setBorder(BorderFactory.createLineBorder(Color.BLACK));
				jpCuandoBarrido.add(jpa);
				
				//Boton para pedir barrido
				jbPidePunto=new JButton("Pide Distancia");
				jbPidePunto.setActionCommand(comandoBotonPideBarrido);
				jbPidePunto.addActionListener(this);
				jpCuandoBarrido.add(jbPidePunto);
				
				jlEsperando=new JLabel("Esperando datos");
				jlEsperando.setEnabled(false);
				jpCuandoBarrido.add(jlEsperando);

				jpSur.add(jpCuandoBarrido);
			}
						
			
			add(jpSur,BorderLayout.SOUTH);
			
			setSize(550, 100);
			setResizable(false);
			setVisible(true);
		}

		/** 
		 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
		 */
		public void actionPerformed(ActionEvent e) {
			if(comandoBotonPideBarrido.equals(e.getActionCommand())) {
				AgenteMuestraPunto.this.removeBehaviour(compEscaneoContinuo);
				compPideBarrido.reset();
				AgenteMuestraPunto.this.addBehaviour(compPideBarrido);
			} else if(comandoBotonEscaneoContinuo.equals(e.getActionCommand())) {
				if(((JCheckBox)e.getSource()).isSelected()) { 
					jbPidePunto.setEnabled(false);
					compEscaneoContinuo.reset();
					AgenteMuestraPunto.this.addBehaviour(compEscaneoContinuo);
				} else {
					jbPidePunto.setEnabled(true);
					AgenteMuestraPunto.this.removeBehaviour(compEscaneoContinuo);					
				}
			}
		}
		
		/**
		 * establece los datos seún punto pasado y programa actualización
		 * @param pto punto obtenido
		 */
		public void actualizaPto(GPSData pto) {
			if(pto==null) {
				//no están las 3 distancias, regresamos
				jlLat.setEnabled(false);
				jlLon.setEnabled(false);
				jlAltura.setEnabled(false);
			} else {
				jlLat.setText(String.format("%2.3f", pto.getLatitud()));
				jlLon.setText(String.format("%2.3f", pto.getLongitud()));
				jlAltura.setText(String.format("%2.3f", pto.getAltura()));
			}
			//programamos la actualizacion de la ventana
			SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					jlLat.repaint();						
					jlLon.repaint();						
					jlAltura.repaint();						
				}
			});

		}

		
		/**
		 * Lo invoca el agente cuando lanza petición de datos
		 */
		public void comienzaEscaneo() {
			if(cbEscaneo.isSelected()) {
				//estamos en escaneo continuo
				//Marcamos estamos esperando
				jlEsperando.setEnabled(true);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						jlEsperando.repaint();						
					}
				});
			} else {
				//Fue petición del botón. debemos activar etiquieta y desactivar botón
				jlEsperando.setEnabled(true);
				jbPidePunto.setEnabled(false);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						jlEsperando.repaint();
						jbPidePunto.repaint();
					}
				});
			}
		}
		
		/**
		 * Lo invoca el agente cuando termina la petición de datos.
		 */
		public void finalizaEscaneo() {
			if(cbEscaneo.isSelected()) {
				//estamos en escaneo continuo
				//Desactivamos estamos esperando
				jlEsperando.setEnabled(false);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						jlEsperando.repaint();						
					}
				});
			} else {
				//Fue petición del botón. debemos activar etiquieta y desactivar botón
				jlEsperando.setEnabled(false);
				jbPidePunto.setEnabled(true);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						jlEsperando.repaint();
						jbPidePunto.repaint();
					}
				});
			}
			
		}

		/* (non-Javadoc)
		 * @see javax.swing.event.ChangeListener#stateChanged(javax.swing.event.ChangeEvent)
		 */
		public void stateChanged(ChangeEvent e) {
			// TODO Auto-generated method stub
			
		}
	}  //fin de agenteMPGUI
	
	//Clases privadas con los comportamientos
	 
	/**
	 * Comportamiento ciclico para iniciar la peticion del barrido cada cierto tiempo.
	 * 
	 * @author alberto
	 * 
	 */
	private class ComportaEscaneoContinuo extends CyclicBehaviour {
		/** Comportamiento que se reutilizará en cada iteración */
		private ComportaPidePto rb=null;
		/** Tiempo de espera entre cada iteración */
		private static final long esperaMs=1000;
		/** Amacena milisegundo en que debe ser la siguiente iteración */
		private long siguiente=0; 

		/**
		 * Añade el comportamiento {@link ComportaPidePto} si se ha cumplide el tiempo.
		 * <p>
		 * Se controla que el tiempo de la siguiente iteración se haya alcanzado. 
		 * Si no es así se bloquea por el tiempo restante. 
		 * Esto debe ser así porque el comportamiento se activa cada vez que llega mensaje.
		 */
		public void action() {
			long ct=System.currentTimeMillis();
			//si no ha llegado nuestro tiempo nos bloqueamos otra vez
			if (ct<siguiente) {
				agentLog("Faltan "+(siguiente-ct)+" ms");
				block(siguiente-ct);
				return;
			}

			//Lanzamos petición si tenemos referencia al agente
			if (agenteGPS()!=null) {
				agentLog("Entramos accion PideBarrido");
				if(rb==null) {
					rb=new ComportaPidePto();
					addBehaviour(rb);
					agentLog("Añadimos el comportamiento RequestBarrido");
				} else {
					if(rb.done()) {
						//Lanzamos nuevamente
						rb.reset();
						addBehaviour(rb);
					} else {
						agentLog("RequestBarrido no ha terminado, no lo lanzamos");					
					}
				}
			}

			//esperamos según se indica en el GUI
			siguiente=System.currentTimeMillis() + miGui.modelFrecEscaneo.getNumber().longValue();
			block(esperaMs);
		}
		
	}  //The la clase ComportaEscaneoContinuo

	/**
	 * Implementa conversación para la recepción del punto.
	 * Si se recibe correctamente barrido lo actualiza en la ventana gráfica
	 * @author alberto
	 */
	private class ComportaPidePto extends SimpleAchieveREInitiator {
		/** Lleva número de petición actual */
		private int NumPeticion=0;
		private Pattern patronContenido; 

		/** Prepara protocolo para realizar <i>QUERY_REF</i> */
		public ComportaPidePto() {
			super(AgenteMuestraPunto.this,new ACLMessage(ACLMessage.QUERY_REF));
		}

		
		/**
		 * Si se resetea indica escaneo finalizado
		 * @see jade.proto.SimpleAchieveREInitiator#reset()
		 */
		@Override
		public void reset() {
			super.reset();
			miGui.finalizaEscaneo(); //por si había algún escaneo pendiente
		}


		@Override
		/**
		 * Contenido de mensaje de petición es <i>Manda imagen</i>.
		 */
		protected ACLMessage prepareRequest(ACLMessage msg) {
			NumPeticion++; //nueva petición
			agentLog(NumPeticion +" Preparamos mensaje a mandar");
			if (msg==null){
//				agentLog("El mensaje es Null :-(((");
				msg=new ACLMessage(ACLMessage.QUERY_REF);
			}
			if(agenteGPS()==null)
				return	null;	//si no hay destinatarios, no mandamos nada
			msg.addReceiver(agenteGPS());
			msg.setOntology("GPS");
			msg.setProtocol(FIPANames.InteractionProtocol.FIPA_QUERY);
			msg.setContent("(UtimoPunto)");
			agentLog("Enviamos mensaje con contenido "+msg.getContent());
			miGui.comienzaEscaneo();
			return msg;
		}

		@Override
		/**
		 * Actulizamos el GUI con la imagen recibida en el mensaje.
		 * <p>
		 * Solo se reciben mensaje <i>INFORM</i> 
		 */
		protected void handleInform(ACLMessage msg) {
			agentLog(NumPeticion +" recibimos el mensaje con la informacion");
			//Pasamos a interpretar el contenido
			try {
				Object om= msg.getContentObject();
				agentLog("El objeto recibido es de la clase:" +
						om.getClass().getName());
				if(! (om instanceof GPSData)) {
					agentLog("El objeto No puede ser tratado porque no es "+
							GPSData.class.getName()
					);
				} else {
					miGui.actualizaPto((GPSData)om);
					miGui.finalizaEscaneo();
				}
			} catch (UnreadableException e) {
				agentLog("No se pudo leer el objeto en mensaje por "+ e.getMessage());
				e.printStackTrace();				
				miGui.actualizaPto(null);
				miGui.finalizaEscaneo();
			}
		}

		/** 
		 * Indicamos que ha habido fallo
		 * @see jade.proto.SimpleAchieveREInitiator#handleFailure(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleFailure(ACLMessage msg) {
			agentLog("("+getClass().getName()+") Mensaje de fallo:"+msg.getContent());
			miGui.actualizaPto(null);
			miGui.finalizaEscaneo(); //por si había algún escaneo pendiente
		}

		/**
		 * Indicamos mensaje no entendido
		 * @see jade.proto.SimpleAchieveREInitiator#handleNotUnderstood(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleNotUnderstood(ACLMessage msg) {
			agentLog("("+getClass().getName()+") Nuestro mensaje no fue entendido:"+msg.getContent());
			miGui.actualizaPto(null);
			miGui.finalizaEscaneo(); //por si había algún escaneo pendiente
		}

		/**
		 * Indicamos el agente no lo quiere hacer
		 * @see jade.proto.SimpleAchieveREInitiator#handleRefuse(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleRefuse(ACLMessage msg) {
			agentLog("("+getClass().getName()+") El interlocutor no lo quiere hacer:"+msg.getContent());
			miGui.actualizaPto(null);
			miGui.finalizaEscaneo(); //por si había algún escaneo pendiente
		}

	}   // de la clase ComportaPideBarrido
}
