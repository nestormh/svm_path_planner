package sibtra.gps;

import javax.comm.*;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.TooManyListenersException;
import java.io.IOException;

public class SimpleConnection {

    private boolean open;

    private SerialParameters parameters;
    private OutputStream os;

    private CommPortIdentifier portId;
    private SerialPort sPort;

    /**
    Creates a SerialConnection object and initilizes variables passed in
    as params.

    @param parameters A SerialParameters object.
    */
    public SimpleConnection( SerialParameters parameters) {

        this.parameters = parameters;
        open = false;
        try {
            openConnection();
          } catch (SerialConnectionException e2) {

              System.out.println("Error al abrir el puerto");
              System.out.flush();
              System.exit(1);
       }
       if (isOpen())
           System.out.println("Puerto Abierto");

   }

      public SimpleConnection(String portName) {



          parameters = new SerialParameters(portName, 9600, 0, 0, 8, 1, 0);
          try {
            openConnection();
          } catch (SerialConnectionException e2) {

              System.out.println("Error al abrir el puerto " + portName);
              System.out.flush();
              System.exit(1);
       }
       if (isOpen())
           System.out.println("Puerto Abierto BaudRate "+ portName);

    }

   public SimpleConnection(String portName,
                              int baudRate,
                              int flowControlIn,
                              int flowControlOut,
                              int databits,
                              int stopbits,
                              int parity) {



          parameters = new SerialParameters(portName, baudRate, flowControlIn, flowControlOut, databits, stopbits, parity);
          try {
            openConnection();
          } catch (SerialConnectionException e2) {

              System.out.println("Error al abrir el puerto " + portName);
              System.out.flush();

       }
       if (isOpen())
           System.out.println("Puerto Abierto BaudRate "+ baudRate);

    }

    /**
    Attempts to open a serial connection and streams using the parameters
    in the SerialParameters object. If it is unsuccesfull at any step it
    returns the port to a closed state, throws a
    <code>SerialConnectionException</code>, and returns.

    Gives a timeout of 30 seconds on the portOpen to allow other applications
    to reliquish the port if have it open and no longer need it.
    */
    public void openConnection() throws SerialConnectionException {

         // Obtain a CommPortIdentifier object for the port you want to open.
         try {
             portId = CommPortIdentifier.getPortIdentifier(parameters.getPortName());
         } catch (NoSuchPortException e) {
             throw new SerialConnectionException(e.getMessage());
         }

         // Open the port represented by the CommPortIdentifier object. Give
         // the open call a relatively long timeout of 30 seconds to allow
         // a different application to reliquish the port if the user
         // wants to.
         try {
             sPort = (SerialPort)portId.open("SerialCam", 30000);
         } catch (PortInUseException e) {
             throw new SerialConnectionException(e.getMessage());
         }

         // Set the parameters of the connection. If they won't set, close the
         // port before throwing an exception.
         try {
             setConnectionParameters();
         } catch (SerialConnectionException e) {
             sPort.close();
             throw e;
         }

         // Open the input and output streams for the connection. If they won't
         // open, close the port before throwing an exception.
         try {
             os = sPort.getOutputStream();
         } catch (IOException e) {
             sPort.close();
             throw new SerialConnectionException("Error opening i/o streams");
         }

         open = true;

         sPort.disableReceiveTimeout();
    }

    public void setConnectionParameters() throws SerialConnectionException {

        // Save state of parameters before trying a set.
        int oldBaudRate = sPort.getBaudRate();
        int oldDatabits = sPort.getDataBits();
        int oldStopbits = sPort.getStopBits();
        int oldParity   = sPort.getParity();
        int oldFlowControl = sPort.getFlowControlMode();

        // Set connection parameters, if set fails return parameters object
        // to original state.
        try {
            sPort.setSerialPortParams(parameters.getBaudRate(),
                                      parameters.getDatabits(),
                                      parameters.getStopbits(),
                                      parameters.getParity());
        } catch (UnsupportedCommOperationException e) {
            parameters.setBaudRate(oldBaudRate);
            parameters.setDatabits(oldDatabits);
            parameters.setStopbits(oldStopbits);
            parameters.setParity(oldParity);
            throw new SerialConnectionException("Unsupported parameter");
        }

        // Set flow control.
        try {
            sPort.setFlowControlMode(parameters.getFlowControlIn()
                                   | parameters.getFlowControlOut());
        } catch (UnsupportedCommOperationException e) {
            throw new SerialConnectionException("Unsupported flow control");
        }
    }

    /**
    Close the port and clean up associated elements.
    */
    public void closeConnection() {
        // If port is alread closed just return.
        if (!open) {
            return;
        }

        // Remove the key listener.
//	messageAreaOut.removeKeyListener(keyHandler);

        // Check to make sure sPort has reference to avoid a NPE.
        if (sPort != null) {
            try {
                // close the i/o streams.
                os.close();
            } catch (IOException e) {
                System.err.println(e);
            }

            // Close the port.
            sPort.close();
        }

        open = false;
    }

    /**
    Send a one second break signal.
    */
    public void sendBreak() {
        sPort.sendBreak(1000);
    }

    /**
    Reports the open status of the port.
    @return true if port is open, false if port is closed.
    */
    public boolean isOpen() {
        return open;
    }

    public void Envia(int a[]) {
        for (int i = 0; i < a.length; i++)
            try {
               os.write(a[i]);

            } catch (Exception e){
               System.out.println("Error al enviar");
            };
    }
}
